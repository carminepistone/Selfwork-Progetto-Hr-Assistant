import chromadb
from chromadb.utils import embedding_functions
from config import Config

class Database:
    def __init__(self):
     
        self.local_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=Config.OPENAI_API_KEY,
            model_name=Config.EMBEDDING_MODEL
        )
        
        self.client = chromadb.PersistentClient(path=Config.PERSISTENT_DIR)
        
        
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