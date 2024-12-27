import os
import toml

# Load the configuration
config = toml.load("config.toml")

# Access the API keys
openai_key = config["API_KEYS"]["OPENAI"]

# Set the environment variable for OpenAI API key
os.environ["OPENAI_API_KEY"] = openai_key

# Create and run the agent
def main():
    return

if __name__ == "__main__":
    main()
