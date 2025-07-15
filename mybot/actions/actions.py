import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3"
BASE_URL = os.getenv("BASE_URL", "http://google.com/")



