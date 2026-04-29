from openai import OpenAI
import os
import uuid
from config import Config

class DocumentProcessor:
    @staticmethod
    def extract_info_with_llm(text_preview):
        """Estrae info anagrafiche una tantum usando OpenAI"""
        client = OpenAI(api_key=Config.OPENAI_API_KEY)
        prompt = f"""Estrai Nome, Email e Telefono da questo incipit di CV. 
        Rispondi ESCLUSIVAMENTE in formato testuale: Nome: ... | Email: ... | Tel: ...
        Testo: {text_preview}"""
        
        response = client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    @staticmethod
    def process_documents():
        documents, metadatas, ids = [], [], []

        for filename in os.listdir(Config.DOCUMENTS_DIR):
            if filename.endswith(".txt"):
                with open(os.path.join(Config.DOCUMENTS_DIR, filename), 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    info_line = DocumentProcessor.extract_info_with_llm(content[:500])
                    
                    
                    documents.append(content)
                    metadatas.append({
                        "source": filename,
                        "candidate_info": info_line
                    })
                    ids.append(str(uuid.uuid4()))

        return documents, metadatas, ids