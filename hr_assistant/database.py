import chromadb
from chromadb.utils import embedding_functions
from config import Config


class Database:

    def __init__(self):
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=getattr(Config, "OPENAI_API_KEY", getattr(Config, "OPENAI_KEY", None)),
            model_name=getattr(Config, "EMBEDDING_MODEL", getattr(Config, "MODEL_NAME", "text-embedding-3-small")),
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
        """Elimina completamente la collection e la reinizializza."""
        try:
            self.client.delete_collection(Config.COLLECTION_NAME)
        except Exception:
            pass
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

    def query(self, query_text: str, n_results: int = 3):
        try:
            return self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
            )
        except Exception as e:
            print(f"Query error: {e}")
            return {}

    @staticmethod
    def get_candidate_name_from_results(results: dict) -> str:
        """
        Estrae il nome del candidato dai metadati dei chunk restituiti da una query.
        Usa il primo chunk disponibile — il nome è lo stesso per tutti i chunk
        dello stesso documento.

        Args:
            results: dizionario restituito da db.query()

        Returns:
            Nome del candidato o 'Candidato Sconosciuto' come fallback
        """
        try:
            metadatas = results.get("metadatas", [])
            if metadatas and metadatas[0]:
                name = metadatas[0][0].get("candidate_name", "").strip()
                if name:
                    return name
        except Exception as e:
            print(f"[WARNING] Impossibile leggere candidate_name dai metadati: {e}")

        return "Candidato Sconosciuto"


    def get_tracked_files(self):
        """Restituisce file unici con metadati."""
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
                        "candidate_name": metadata.get("candidate_name", "Candidato Sconosciuto"),
                    }

        return tracked_files

    def remove_document_by_source(self, source: str):
        """Rimuove tutti i chunk associati a un file."""
        try:
            result = self.collection.get(where={"source": source})
            if result and result.get("ids"):
                self.collection.delete(ids=result["ids"])
        except Exception as e:
            print(f"Remove error ({source}): {e}")


    def get_stats(self) -> str:
        try:
            result = self.collection.get()

            if not result or not result.get("metadatas"):
                return "Database vuoto"

            unique_files = {}
            for m in result["metadatas"]:
                source = m.get("source")
                if source and source not in unique_files:
                    unique_files[source] = m.get("candidate_name", "Candidato Sconosciuto")

            file_lines = "\n".join(
                f"  - {source} ({name})"
                for source, name in unique_files.items()
            )

            return (
                f"\nNome Collezione: {self.collection.name}\n"
                f"Numero totale Frammenti: {self.collection.count()}\n"
                f"Numero Files Elaborati: {len(unique_files)}\n"
                f"Candidati indicizzati:\n{file_lines}\n"
            )

        except Exception as e:
            return f"Errore lettura statistiche: {str(e)}"
