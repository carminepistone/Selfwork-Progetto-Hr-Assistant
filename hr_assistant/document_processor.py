import ollama
import os
import uuid
from config import Config
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    @staticmethod
    def extract_info_with_llm(text_preview):
        """Estrae info anagrafiche una tantum usando Llama 3.1"""
        prompt = f"""Estrai Nome, Email e Telefono da questo incipit di CV. 
        Rispondi ESCLUSIVAMENTE in formato testuale: Nome: ... | Email: ... | Tel: ...
        Testo: {text_preview}"""
        
        response = ollama.chat(model=Config.LLM_MODEL, messages=[{"role": "user", "content": prompt}])
        return response['message']['content']

    @staticmethod
    def process_documents():
        documents, metadatas, ids = [], [], []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        for filename in os.listdir(Config.DOCUMENTS_DIR):
            if filename.endswith(".txt"):
                with open(os.path.join(Config.DOCUMENTS_DIR, filename), 'r', encoding='utf-8') as f:
                    content = f.read()
                    

                    info_line = DocumentProcessor.extract_info_with_llm(content[:500])
                    
                    # 2. SPLITTING
                    chunks = text_splitter.split_text(content)
                    for chunk in chunks:
                        documents.append(chunk)

                        metadatas.append({
                            "source": filename,
                            "candidate_info": info_line
                        })
                        ids.append(str(uuid.uuid4()))
        return documents, metadatas, ids