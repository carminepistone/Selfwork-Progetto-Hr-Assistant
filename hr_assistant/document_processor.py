import os
import uuid
import hashlib
from openai import OpenAI
from langchain_text_splitters import CharacterTextSplitter
from config import Config


class DocumentProcessor:

    @staticmethod
    def read_first_lines(file_path, n_lines=10):
        """Legge le prime N righe del file (usato per estrarre info candidato)"""
        with open(file_path, "r", encoding="utf-8") as file:
            return [line.strip() for line, _ in zip(file, range(n_lines))]

    @staticmethod
    def get_file_hash(file_path):
        """Hash MD5 del contenuto per rilevare modifiche"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @staticmethod
    def get_document_metadata(file_path):
        """Metadati base: hash, last_modified, source"""
        return {
            "hash": DocumentProcessor.get_file_hash(file_path),
            "last_modified": os.path.getmtime(file_path),
            "source": os.path.basename(file_path),
        }

    @staticmethod
    def extract_info_with_llm(text_preview):
        """Estrae Nome, Email, Telefono dall'incipit del CV tramite LLM"""
        client = OpenAI(api_key=Config.OPENAI_API_KEY)
        prompt = f"""Estrai Nome, Email e Telefono da questo incipit di CV.
        Rispondi ESCLUSIVAMENTE in formato: Nome: ... | Email: ... | Tel: ...
        Testo: {text_preview}"""
        response = client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    @staticmethod
    def process_single_document(file_path):
        """Chunking del documento con LangChain + metadati arricchiti con info LLM"""
        documents, metadatas, ids = [], [], []

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        file_metadata = DocumentProcessor.get_document_metadata(file_path)
        candidate_info = DocumentProcessor.extract_info_with_llm(content[:500])

        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_text(content)

        for chunk in chunks:
            if chunk and not chunk.isspace():
                documents.append(chunk)
                metadatas.append({
                    **file_metadata,
                    "candidate_info": candidate_info
                })
                ids.append(str(uuid.uuid4()))

        return documents, metadatas, ids

    @staticmethod
    def process_documents(db):
        """Sync intelligente: aggiunge, aggiorna e rimuove documenti dal DB"""
        current_files = {
            f: DocumentProcessor.get_document_metadata(
                os.path.join(Config.DOCUMENTS_DIR, f)
            )
            for f in os.listdir(Config.DOCUMENTS_DIR)
            if f.endswith(".txt")
        }
        existing_files = db.get_tracked_files()

        files_to_add = set(current_files.keys()) - set(existing_files.keys())
        files_to_remove = set(existing_files.keys()) - set(current_files.keys())
        files_to_update = {
            f for f in set(current_files.keys()) & set(existing_files.keys())
            if current_files[f]["hash"] != existing_files[f]["hash"]
        }

        for action, files in [("add", files_to_add), ("update", files_to_update)]:
            for filename in files:
                file_path = os.path.join(Config.DOCUMENTS_DIR, filename)
                documents, metadatas, ids = DocumentProcessor.process_single_document(file_path)
                if action == "update":
                    db.remove_document_by_source(filename)
                if documents:
                    db.add_documents(documents, metadatas, ids)

        for filename in files_to_remove:
            db.remove_document_by_source(filename)

        return len(files_to_add), len(files_to_update), len(files_to_remove)