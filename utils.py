import toml
from openai import OpenAI

def create_openai_client() -> OpenAI:
    """
    Creates and returns an OpenAI client using the API key from the configuration file.
    """
    config = toml.load("config.toml")
    api_key = config["API_KEYS"]["OPENAI"]
    if not api_key:
        raise ValueError("OpenAI API key is missing in the configuration file.")
    return OpenAI(api_key=api_key)
