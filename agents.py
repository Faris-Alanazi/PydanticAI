from pydantic_ai import Agent, RunContext
from tools import search_top_k
from utils import create_openai_client
from typing import List, Dict, Union
import toml
import os

# Load API key and set environment variable
config = toml.load("config.toml")
os.environ["OPENAI_API_KEY"] = config["API_KEYS"]["OPENAI"]

# Initialize OpenAI client
client = create_openai_client()

class HistoryEnabledAgent(Agent):
    """
    Custom Agent class with built-in message history management.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message_history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        """
        Add a message to the agent's history.
        """
        self.message_history.append({"role": role, "content": content})

    def clear_history(self):
        """
        Clear the agent's message history.
        """
        self.message_history = []

    def get_history(self) -> List[Dict[str, str]]:
        """
        Retrieve the agent's message history.
        """
        return self.message_history

    async def run_with_history(self, user_input: str) -> Union[str, None]:
        """
        Run the agent with internal message history and return the response.
        """
        # Add user input to history
        self.add_message("user", user_input)

        # Generate the response
        response = await self.run(
            user_input,
            message_history=self.message_history,  # Use built-in history
        )

        # Add response to history
        self.add_message("assistant", response.data)

        return response.data


# Define the custom agent with history enabled
rag_agent = HistoryEnabledAgent(
    model="openai:gpt-4o",
    model_settings={"api_key": os.environ["OPENAI_API_KEY"]},
    system_prompt=(
        "You are a highly skilled coding assistant. When generating code, ensure it is "
        "production-ready, well-documented, and follows best practices. Use the retrieved "
        "context to enhance your response."
    ),
)

@rag_agent.system_prompt
async def dynamic_prompt(ctx: RunContext):
    """
    Create a system prompt dynamically based on query context.
    """
    return (
        "You are a skilled coding assistant. When generating code, ensure it is "
        "production-ready, well-documented, and follows best practices. Use the provided "
        "context to inform your response but do not repeat the context verbatim."
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

    # Step 3: Format combined context for response generation
    return f"""
    User Query: {query}
    Code Context: {combined_text}
    Please generate a Python script or function that satisfies the query using the provided context.
    """
