import json
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from typing import List, Dict, Any
from utils import create_openai_client

cached_embeddings = None

# Initialize OpenAI client
client = create_openai_client()

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    """
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_embedding(text: str, model: str = "text-embedding-3-large") -> List[float]:
    """
    Retrieve the embedding for a given text using the OpenAI API.
    """
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

def load_embeddings(embeddings_file: str) -> List[Dict[str, Any]]:
    """
    Load embeddings from a JSON file with caching.
    """
    global cached_embeddings
    if cached_embeddings is None:
        with open(embeddings_file, 'r') as f:
            cached_embeddings = json.load(f)
    return cached_embeddings

async def search_top_k(query: str, top_k: int) -> List[Dict[str, Any]]:
    """
    Search for the top-k most similar chunks to the query in precomputed embeddings.
    """
    embeddings = load_embeddings("embeddings.json")
    query_embedding = get_embedding(query)

    # Compute similarity
    results = [
        {"chunk": entry["chunk"], "similarity": cosine_similarity(query_embedding, entry["embedding"])}
        for entry in tqdm(embeddings, desc="Comparing embeddings", unit="chunk")
        if "def " in entry["chunk"] or "class " in entry["chunk"]  # Basic filter for code-related chunks
    ]

    # Sort by similarity
    return sorted(results, key=lambda x: x["similarity"], reverse=True)[:top_k]
