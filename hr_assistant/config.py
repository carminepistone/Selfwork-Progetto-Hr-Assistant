import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
    DOCUMENTS_DIR = os.path.join(ROOT_DIR, os.getenv("DOCUMENTS_DIR", "resumes"))
    PERSISTENT_DIR = os.path.join(ROOT_DIR, "data", "chromadb")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "CVs")
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "text-embedding-3-small"
    )


    MODEL_NAME = EMBEDDING_MODEL
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_MODEL_HIGH = os.getenv("LLM_MODEL_HIGH", "gpt-4o")

    AI_API_URL = os.getenv("AI_API_URL", "https://api.openai.com/v1")
    AI_API_KEY = os.getenv("AI_API_KEY") or os.getenv("OPENAI_API_KEY")


    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_KEY = OPENAI_API_KEY


    MODEL_PATH = os.getenv("MODEL_PATH", "")
