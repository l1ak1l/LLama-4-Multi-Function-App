import os
from dotenv import load_dotenv

# API key management and configuration

# Load environment variables from a .env file (if available)
load_dotenv()

def get_openai_key(default_key: str = None) -> str:
    """
    Retrieve the OpenAI API key from the environment or a provided default.
    
    Parameters:
        default_key (str): A fallback API key if the environment variable is not set.
    
    Returns:
        str: The OpenAI API key.
    
    Raises:
        ValueError: If no API key is found.
    """
    api_key = os.environ.get("OPENAI_API_KEY", default_key)
    if not api_key:
        raise ValueError("OpenAI API key is not set. Please set it in your environment or .env file.")
    return api_key
