import os
import uuid
import hashlib
import mimetypes
import tempfile
import asyncio

from typing import Tuple, List, Dict, Any
from zipfile import ZipFile

from config import Config
from semantic_chunking import SemanticChunking
from markitdown import MarkItDown


class DocumentProcessor:

    SUPPORTED_EXTENSIONS = {
        ".txt": "text",
        ".pdf": "document",
        ".doc": "document",
        ".docx": "document",
        ".ppt": "presentation",
        ".pptx": "presentation",
        ".xls": "spreadsheet",
        ".xlsx": "spreadsheet",
        ".html": "web",
        ".htm": "web",
        ".csv": "data",
        ".json": "data",
        ".xml": "data",
        ".zip": "archive",
    }

    def __init__(self):
        self.md_converter = MarkItDown()

    @staticmethod
    def read_first_lines(file_path: str, n_lines: int = 50) -> List[str]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return [line.strip() for line, _ in zip(f, range(n_lines))]
        except Exception:
            return []

    @staticmethod
    def get_file_hash(file_path: str) -> str:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def get_document_metadata(self, file_path: str) -> Dict[str, Any]:
        ext = os.path.splitext(file_path)[1].lower()
        return {
            "hash": self.get_file_hash(file_path),
            "last_modified": os.path.getmtime(file_path),
            "source": os.path.basename(file_path),
            "file_type": self.SUPPORTED_EXTENSIONS.get(ext, "unknown"),
            "mime_type": mimetypes.guess_type(file_path)[0],
            "extension": ext,
        }

    def _convert_to_markdown(self, file_path: str) -> str:
        try:
            result = self.md_converter.convert(file_path)

            if not result or not hasattr(result, "text_content"):
                print(f"[ERROR] Conversione nulla: {file_path}")
                return ""

            content = result.text_content.strip()

            if not content:
                print(f"[WARNING] Contenuto vuoto dopo conversione: {file_path}")
            else:
                print(f"[OK] {file_path} → {len(content)} chars")

            return content

        except Exception as e:
            print(f"[ERROR] Conversione fallita: {file_path} | {e}")
            return ""

    def _process_zip_file(self, file_path: str) -> str:
        content = ""

        with tempfile.TemporaryDirectory() as temp_dir:
            with ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            for root, _, files in os.walk(temp_dir):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()

                    if ext in self.SUPPORTED_EXTENSIONS:
                        full_path = os.path.join(root, file)
                        extracted = self._convert_to_markdown(full_path)

                        if extracted:
                            content += f"\n\nFile: {file}\n{extracted}"

        return content


    async def process_single_document(
        self, file_path: str
    ) -> Tuple[List[str], List[Dict], List[str]]:
        from utils import LLMHelper

        print(f"\n[PROCESSING] {file_path}")

        documents, metadatas, ids = [], [], []

        ext = os.path.splitext(file_path)[1].lower()
        file_type = self.SUPPORTED_EXTENSIONS.get(ext)

        if not file_type:
            print("[SKIP] Unsupported file")
            return documents, metadatas, ids

        if file_type == "archive":
            content = self._process_zip_file(file_path)

        elif file_type == "text":
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                print(f"[ERROR TXT] {e}")
                content = ""

        else:
            content = self._convert_to_markdown(file_path)

        print(f"[DEBUG] Content length: {len(content)}")

        if not content:
            print("[SKIP] Documento ignorato (contenuto vuoto)")
            return documents, metadatas, ids

        header_text = "\n".join(content.splitlines()[:50])
        candidate_name = await LLMHelper.get_candidate_name(header_text)
        print(f"[CANDIDATE] {candidate_name}")

        sc = SemanticChunking(
            breakpoint_percentile=65,
            buffer_size=3,
        )
        chunks = sc.chunk_text(content)

        print(f"[DEBUG] Chunks: {len(chunks)}")

        if not chunks:
            print("[SKIP] Nessun chunk generato")
            return documents, metadatas, ids

        metadata = self.get_document_metadata(file_path)
        metadata["candidate_name"] = candidate_name 

        for chunk in chunks:
            if chunk and not chunk.isspace():
                documents.append(chunk)
                metadatas.append(metadata)
                ids.append(str(uuid.uuid4()))

        print(f"[OK] Salvati {len(documents)} chunk per '{candidate_name}'")

        return documents, metadatas, ids


    async def process_documents(self, db) -> Tuple[int, int, int]:
        print("\n[SCAN DIR]", Config.DOCUMENTS_DIR)

        current_files = {
            f: self.get_document_metadata(os.path.join(Config.DOCUMENTS_DIR, f))
            for f in os.listdir(Config.DOCUMENTS_DIR)
            if os.path.splitext(f)[1].lower() in self.SUPPORTED_EXTENSIONS
        }

        print("[FILES]", list(current_files.keys()))

        existing_files = db.get_tracked_files()

        files_to_add = set(current_files) - set(existing_files)
        files_to_remove = set(existing_files) - set(current_files)

        files_to_update = {
            f for f in current_files.keys() & existing_files.keys()
            if current_files[f]["hash"] != existing_files[f]["hash"]
        }

        for action, files in [("add", files_to_add), ("update", files_to_update)]:
            for filename in files:
                path = os.path.join(Config.DOCUMENTS_DIR, filename)

                docs, metas, ids = await self.process_single_document(path)

                if action == "update":
                    db.remove_document_by_source(filename)

                if docs:
                    db.add_documents(docs, metas, ids)

        for filename in files_to_remove:
            db.remove_document_by_source(filename)

        return len(files_to_add), len(files_to_update), len(files_to_remove)