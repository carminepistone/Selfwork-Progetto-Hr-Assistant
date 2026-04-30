import chromadb
from chromadb.utils import embedding_functions
from config import Config


class Database:
    def __init__(self):

        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=getattr(Config, "OPENAI_API_KEY", getattr(Config, "OPENAI_KEY")),
            model_name=getattr(Config, "EMBEDDING_MODEL", getattr(Config, "MODEL_NAME")),
        )

        self.client = chromadb.PersistentClient(path=Config.PERSISTENT_DIR)
        self._init_collection()


    def _init_collection(self):
        self.collection = self.client.get_or_create_collection(
            name=Config.COLLECTION_NAME,
            embedding_function=self.openai_ef,
            metadata={"hnsw:space": "cosine"},
        )

    def delete_collection(self):
        """Elimina completamente la collection e la reinizializza"""
        try:
            self.client.delete_collection(Config.COLLECTION_NAME)
        except Exception:
            pass  # già inesistente

        self._init_collection()


    def add_documents(self, documents, metadatas, ids):
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
        except Exception as e:
            print(f"Add warning: {e}")

    def query(self, query_text, n_results=3):
        try:
            return self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
            )
        except Exception as e:
            print(f"Query error: {e}")
            return {}


    # Tracciamento file

    def get_tracked_files(self):
        """Restituisce file unici con metadati"""
        result = self.collection.get()
        tracked_files = {}

        if result and result.get("metadatas"):
            for metadata in result["metadatas"]:
                source = metadata.get("source")
                if source and source not in tracked_files:
                    tracked_files[source] = {
                        "hash": metadata.get("hash"),
                        "last_modified": metadata.get("last_modified"),
                        "source": source,
                    }

        return tracked_files

    def remove_document_by_source(self, source):
        """Rimuove tutti i chunk associati a un file"""
        try:
            result = self.collection.get(where={"source": source})
            if result and result.get("ids"):
                self.collection.delete(ids=result["ids"])
        except Exception as e:
            print(f"Remove error ({source}): {e}")


    # Statistiche

    def get_stats(self):
        try:
            result = self.collection.get()

            if not result or not result.get("metadatas"):
                return "Database vuoto"

            unique_files = {
                m["source"] for m in result["metadatas"] if "source" in m
            }

            return f"""
Nome Collezione: {self.collection.name}
Numero totale Frammenti: {self.collection.count()}
Numero Files Elaborati: {len(unique_files)}
"""
        except Exception as e:
            return f"Errore lettura statistiche: {str(e)}"