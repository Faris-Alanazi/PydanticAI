from pydantic_ai import Agent
from tools import search_top_k
from utils import create_openai_client
from typing import List
import toml
import os

# Load API key and set environment variable
config = toml.load("config.toml")
os.environ["OPENAI_API_KEY"] = config["API_KEYS"]["OPENAI"]

# Initialize OpenAI client
client = create_openai_client()

# Define the agent
rag_agent = Agent(
    model="openai:gpt-4o",
    model_settings={"api_key": os.environ["OPENAI_API_KEY"]},
    system_prompt=(
        "You are a knowledgeable assistant with the ability to retrieve and analyze information. "
        "The information you need to find is stored in a vector database, which is originally "
        "derived from a GitHub repository. Use this knowledge to tailor your search queries "
        "so you can fetch the most relevant and precise information needed to address user queries."
    ),
)

@rag_agent.tool_plain
async def rag_search(query: str) -> str:
    """
    Perform a RAG search on data originally from a GitHub repository, 
    analyze the results, and answer the user's query.
    """
    # Step 1: Perform the RAG search
    query_length = len(query.split())
    top_k = max(2, min(5, query_length // 3))
    results = await search_top_k(query, top_k)

    # Step 2: Analyze the results
    if not results:
        return "I couldn't find relevant information to answer your query."

    # Combine results for better context
    combined_text = " ".join([result["chunk"] for result in results])

    # Step 3: Use OpenAI chat model to generate the answer
    messages = [
        {"role": "system", "content": (
            "You are a knowledgeable assistant. The context of your search results is originally derived "
            "from a GitHub repository before being transformed into a vector format. Use the provided context "
            "to generate an insightful and accurate answer."
        )},
        {"role": "user", "content": f"Context: {combined_text}\n\nUser Query: {query}\n\nAnswer the query based on the context."}
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=500,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()