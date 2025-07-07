# Startup Clustering API

A FastAPI-based service that clusters startups based on their descriptions using OpenAI embeddings and HDBSCAN clustering.

## Features

- Clusters startups based on their descriptions
- Uses OpenAI's text-embedding-3-small model for embeddings
- Implements HDBSCAN clustering algorithm
- RESTful API with automatic documentation

## API Endpoints

### POST /cluster
Clusters a list of startups based on their descriptions.

**Request Body:**
```json
{
  "startups": [
    {
      "name": "Startup Name",
      "description": "Startup description..."
    }
  ]
}
```

**Response:**
```json
{
  "clustered": [
    {
      "name": "Startup Name",
      "description": "Startup description...",
      "cluster_id": 0
    }
  ]
}
```

## Environment Variables

Create a `.env` file with:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`

3. Run the server:
```bash
uvicorn main:app --reload
```

## Deployment on Vercel

1. Install Vercel CLI:
```bash
npm i -g vercel
```

2. Deploy:
```bash
vercel
```

3. Set environment variables in Vercel dashboard:
   - `OPENAI_API_KEY`

The API will be available at your Vercel URL. 