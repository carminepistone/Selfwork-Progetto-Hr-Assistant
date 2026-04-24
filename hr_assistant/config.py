import os
from dotenv import load_dotenv

load_dotenv()


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

class Config:
    
    DOCUMENTS_DIR = os.path.join(ROOT_DIR, "resumes")
    PERSISTENT_DIR = os.path.join(ROOT_DIR, "data", "chromadb")
    
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "CVs")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # API Keys 
    AI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
    AI_API_URL = OLLAMA_BASE_URL