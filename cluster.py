import openai
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from dotenv import load_dotenv
import os
from sklearn.metrics import silhouette_score
from collections import defaultdict

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def embed_texts(texts: list[str], model="text-embedding-3-small") -> list[list[float]]:
    all_embeddings = []
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = openai.embeddings.create(input=batch, model=model)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    return all_embeddings

def find_best_k(X, min_k=2, max_k=10):
    n_samples = len(X)
    # k must be at least 2 and at most n_samples - 1
    max_possible_k = min(max_k, n_samples - 1)
    if n_samples < 2:
        raise ValueError("Need at least 2 samples for clustering")
    best_k = min_k
    best_score = -1
    for k in range(min_k, max_possible_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k = k
            best_score = score
    return best_k

def generate_cluster_label(descriptions):
    # Use OpenAI to summarize the cluster, or use a simple heuristic
    prompt = "Summarize the following startup description into a 1 - 3 word Title:\n" + "\n".join(descriptions)
    response = openai.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.5,
    )
    content = response.choices[0].message.content if response.choices and response.choices[0].message.content else None
    return content.strip() if content else "Cluster"

def cluster_startups(startups: list[dict]) -> list[dict]:
    descriptions = [s["description"] for s in startups]
    embeddings = embed_texts(descriptions)

    X = StandardScaler().fit_transform(embeddings)

    k = find_best_k(X)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)

    for i, s in enumerate(startups):
        s["cluster_id"] = int(labels[i])

    # Group by cluster_id
    clusters = defaultdict(list)
    for s in startups:
        clusters[s["cluster_id"]].append(s["description"])

    # Generate label for each cluster
    cluster_labels = {}
    for cluster_id, descriptions in clusters.items():
        cluster_labels[cluster_id] = generate_cluster_label(descriptions)

    # Assign label to each startup
    for s in startups:
        s["label"] = cluster_labels[s["cluster_id"]]

    # Ensure output dicts have only the expected keys
    output = []
    for s in startups:
        output.append({
            "startupName": s["startupName"],
            "description": s["description"],
            "industry": s["industry"],
            "cluster_id": s["cluster_id"],
            "label": s["label"]
        })
    return output
