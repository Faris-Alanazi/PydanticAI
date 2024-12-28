import asyncio
from agents import rag_agent
import toml
import os

# Load configuration and set environment variable
config = toml.load("config.toml")
os.environ["OPENAI_API_KEY"] = config["API_KEYS"]["OPENAI"]

async def main():
    # Query the agent
    query = "How can I create an agent that delegates tasks to other agents? Write me a script for that."
    answer = await rag_agent.run(query)

    # Print the final answer
    print(f"\nAgent's answer:\n{answer.data}")

if __name__ == "__main__":
    asyncio.run(main())
