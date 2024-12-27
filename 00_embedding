# ============ Import Libraries ============
import os
import json
import toml
import numpy as np
from openai import OpenAI
from tqdm import tqdm  # For progress bar
import tiktoken  # For token calculation

# ============ Load Configuration ============
# Load API key from the `config.toml` file
config = toml.load("config.toml")
openai_key = config["API_KEYS"]["OPENAI"]

# Create an OpenAI client
client = OpenAI(api_key=openai_key)

# ============ Helper Functions ============

def get_embedding(text, model="text-embedding-3-large"):
    """
    Retrieves the embedding for a given text string using the OpenAI API.

    Args:
        text (str): The input text to embed.
        model (str): The embedding model to use.

    Returns:
        List[float]: The embedding vector representing the input text.
    """
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding  # Access using dot notation

def cosine_similarity(vec1, vec2):
    """
    Calculates cosine similarity between two vectors.

    Args:
        vec1 (List[float]): The first vector.
        vec2 (List[float]): The second vector.

    Returns:
        float: The cosine similarity score between the two vectors.
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def num_tokens_from_string(string, encoding_name="cl100k_base"):
    """
    Calculates the number of tokens in a string using the specified tokenizer.

    Args:
        string (str): The input text string.
        encoding_name (str): The tokenizer encoding to use (default is 'cl100k_base').

    Returns:
        int: The number of tokens in the string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

def preprocess_and_embed(file_path, output_file, model="text-embedding-3-large", chunk_size=500):
    """
    Reads a file, chunks the text, generates embeddings for each chunk, and saves the results to a JSON file.

    Args:
        file_path (str): Path to the input text file.
        output_file (str): Path to the output JSON file to save embeddings.
        model (str): The OpenAI embedding model to use.
        chunk_size (int): Character size for each text chunk.

    Returns:
        None
    """
    # ============ Read and Split Text ============
    print(f"Reading and processing the file: {file_path}")
    with open(file_path, 'r') as file:
        text = file.read()

    # Calculate total tokens in the text
    total_tokens = num_tokens_from_string(text)
    print(f"Total tokens in the file: {total_tokens}")

    # Prompt user for confirmation before proceeding
    confirm = input("Do you want to continue processing this file? (y/yes to proceed): ").strip().lower()
    if confirm not in ["y", "yes"]:
        print("Processing aborted.")
        return

    # Split the text into smaller chunks of the specified size
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    print(f"Text split into {len(chunks)} chunks of size {chunk_size} characters.")

    # ============ Generate Embeddings ============
    print("Generating embeddings for each chunk...")
    embeddings = []
    for chunk in tqdm(chunks, desc="Processing chunks", unit="chunk"):
        embeddings.append({"chunk": chunk, "embedding": get_embedding(chunk, model)})

    # ============ Save to File ============
    print(f"Saving embeddings to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(embeddings, f)
    print(f"Embeddings saved successfully to {output_file}.")

# ============ Main Execution ============
if __name__ == "__main__":
    # Define file paths and model
    input_file = "Data/pydantic-pydantic-ai.txt"  # Replace with your actual text file
    output_file = "embeddings.json"          # File to save the embeddings
    embedding_model = "text-embedding-3-large"  # Select the OpenAI embedding model

    # Process and embed the input file
    preprocess_and_embed(input_file, output_file, model=embedding_model)













# =================================================================================
#                                   How to Load
# =================================================================================

# import json
# import numpy as np
# from openai import OpenAI
# from tqdm import tqdm

# # Global variable to store cached embeddings
# cached_embeddings = None

# # Create OpenAI client
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# def cosine_similarity(vec1, vec2):
#     """
#     Calculates cosine similarity between two vectors.

#     Args:
#         vec1 (List[float]): The first vector.
#         vec2 (List[float]): The second vector.

#     Returns:
#         float: The cosine similarity score between the two vectors.
#     """
#     vec1 = np.array(vec1)
#     vec2 = np.array(vec2)
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# def get_embedding(text, model="text-embedding-3-large"):
#     """
#     Retrieves the embedding for a given text string using the OpenAI API.

#     Args:
#         text (str): The input text to embed.
#         model (str): The embedding model to use.

#     Returns:
#         List[float]: The embedding vector representing the input text.
#     """
#     response = client.embeddings.create(input=text, model=model)
#     return response.data[0].embedding  # Correctly using dot notation to access the data

# def load_embeddings(embeddings_file):
#     """
#     Loads embeddings from a JSON file into memory, with caching.

#     Args:
#         embeddings_file (str): Path to the JSON file containing precomputed embeddings.

#     Returns:
#         List[dict]: A list of dictionaries with 'chunk' and 'embedding'.
#     """
#     global cached_embeddings
#     if cached_embeddings is None:
#         print(f"Loading embeddings from {embeddings_file}...")
#         with open(embeddings_file, 'r') as f:
#             cached_embeddings = json.load(f)
#         print("Embeddings loaded and cached.")
#     else:
#         print("Using cached embeddings.")
#     return cached_embeddings

# def search_top_k(query, embeddings_file, model="text-embedding-3-large", top_k=3):
#     """
#     Searches for the top-k most similar chunks to the query in the precomputed embeddings.

#     Args:
#         query (str): The query string.
#         embeddings_file (str): Path to the JSON file containing precomputed embeddings.
#         model (str): The OpenAI embedding model to use for the query.
#         top_k (int): Number of top results to return.

#     Returns:
#         List[dict]: Top-k most similar chunks with their similarity scores.
#     """
#     # Load embeddings (cached)
#     embeddings = load_embeddings(embeddings_file)

#     # Generate embedding for the query
#     print("Generating embedding for the query...")
#     query_embedding = get_embedding(query, model=model)

#     # Calculate similarity scores for each chunk
#     print("Calculating similarity scores...")
#     results = []
#     for entry in tqdm(embeddings, desc="Comparing embeddings", unit="chunk"):
#         similarity = cosine_similarity(query_embedding, entry["embedding"])
#         results.append({"chunk": entry["chunk"], "similarity": similarity})

#     # Sort results by similarity and return the top-k
#     top_results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:top_k]
#     return top_results

# # Example Usage
# if __name__ == "__main__":
#     # Define inputs
#     query = "Explain the ToolPrepareFunc in detail."
#     embeddings_file = "embeddings.json"  # Path to the precomputed embeddings file
#     embedding_model = "text-embedding-3-large"  # Model to use for query embedding

#     # Perform the search
#     top_results = search_top_k(query, embeddings_file, model=embedding_model, top_k=3)

#     # Display the top 3 results
#     print("\nTop 3 Results:")
#     for idx, result in enumerate(top_results, start=1):
#         print(f"\nResult {idx}:")
#         print(f"Similarity Score: {result['similarity']:.4f}")
#         print(f"Chunk: {result['chunk']}")

