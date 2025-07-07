from fastapi import FastAPI
from model import ClusterRequest, ClusterResponse, ClusteredStartup
from cluster import cluster_startups
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Startup Clustering API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/cluster", response_model=ClusterResponse)
def cluster_startups_endpoint(data: ClusterRequest):
    input_data = [s.dict() for s in data.startups]
    clustered = cluster_startups(input_data)
    return {"clustered": [ClusteredStartup(**c) for c in clustered]}
