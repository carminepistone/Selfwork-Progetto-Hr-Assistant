import os
from dotenv import load_dotenv
load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

class Config:
    DOCUMENTS_DIR = os.path.join(ROOT_DIR, "resumes")
    PERSISTENT_DIR = os.path.join(ROOT_DIR, "data", "chromadb")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "CVs")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Alias richiesti da utils.py
    AI_API_URL = os.getenv("AI_API_URL", "https://api.openai.com/v1")
    AI_API_KEY = os.getenv("AI_API_KEY", os.getenv("OPENAI_API_KEY"))
    OPENAI_KEY = os.getenv("OPENAI_API_KEY") 
    MODEL_NAME = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")