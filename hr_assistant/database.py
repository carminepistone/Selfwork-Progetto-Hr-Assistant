import chromadb
from chromadb.utils import embedding_functions
from config import Config

class Database:
    def __init__(self):
        # Sostituzione con Ollama Embedding Function
        self.local_ef = embedding_functions.OllamaEmbeddingFunction(
            url=f"{Config.OLLAMA_BASE_URL}/api/embeddings",
            model_name=Config.EMBEDDING_MODEL
        )

        # Inizializzazione client persistente
        self.client = chromadb.PersistentClient(path=Config.PERSISTENT_DIR)
        
        # Creazione o recupero collezione con la nuova funzione di embedding
        self.collection = self.client.get_or_create_collection(
            name=Config.COLLECTION_NAME, 
            embedding_function=self.local_ef
        )

    def add_documents(self, documents, metadatas, ids):
        try:
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        except Exception as e:
            print(f"Nota: Alcuni documenti potrebbero essere già presenti: {e}")

    def query(self, query_text, n_results=1):
        return self.collection.query(query_texts=[query_text], n_results=n_results)