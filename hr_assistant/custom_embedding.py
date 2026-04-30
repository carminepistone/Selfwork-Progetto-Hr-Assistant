import os
from chromadb.api.types import EmbeddingFunction
from chromadb.utils import embedding_functions
import openai
from config import Config

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import ollama
except ImportError:
    ollama = None


class CustomEmbeddingFunction(EmbeddingFunction):
    """
    Funzione di embedding compatibile con ChromaDB.
    Supporta: openai, local (SentenceTransformer), ollama.
    Il provider si configura tramite Config.EMBEDDING_PROVIDER.
    """

    def __init__(self):
        self.provider = Config.EMBEDDING_PROVIDER
        self.model_name = Config.MODEL_NAME
        self.model_path = Config.MODEL_PATH

        if self.provider == "openai":
            self._setup_openai()
        elif self.provider == "local":
            self._setup_local_model()
        elif self.provider == "ollama":
            self._setup_ollama()
        else:
            raise ValueError(f"EMBEDDING_PROVIDER '{self.provider}' non supportato.")

    def _setup_openai(self):
        openai.api_key = Config.OPENAI_API_KEY
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=Config.OPENAI_API_KEY,
            model_name=self.model_name
        )

    def _setup_local_model(self):
        if SentenceTransformer is None:
            raise ImportError("Installa sentence-transformers: pip install sentence-transformers")
        if os.path.exists(self.model_path):
            self.embedding_function = SentenceTransformer(self.model_path)
        else:
            self.embedding_function = SentenceTransformer(self.model_name)
            self.embedding_function.save_pretrained(self.model_path)

    def _setup_ollama(self):
        if ollama is None:
            raise ImportError("Installa ollama: pip install ollama")

    def __call__(self, texts):
        if self.provider == "openai":
            return self.embedding_function(texts)
        if self.provider == "local":
            return self.embedding_function.encode(texts).tolist()
        if self.provider == "ollama":
            return [
                ollama.embeddings(model=self.model_name, prompt=text)["embedding"]
                for text in texts
            ]