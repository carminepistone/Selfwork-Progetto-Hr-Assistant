import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import os
import chainlit as cl
from document_processor import DocumentProcessor
from database import Database
from config import Config
from utils import LLMHelper


db = Database()
if not os.path.exists("./chroma_db"): 
    documents, metadatas, ids = DocumentProcessor.process_documents()
    db.add_documents(documents, metadatas, ids)

@cl.on_chat_start
def start():
    cl.user_session.set(
        "messages",
        [
            {
                "role": "system",
                "content": """Sei un Senior HR Recruiter esperto in analisi tecnica. 
                Il tuo compito è analizzare i CV forniti e confrontarli con i requisiti della posizione.
                Sii oggettivo: evidenzia punti di forza, lacune (gap) e potenziale del candidato.
                Rispondi in modo conciso utilizzando elenchi puntati per la leggibilità.""",
            }
        ],
    )

@cl.on_message
async def handle_message(message: cl.Message):
    user_question = message.content
    results = db.query(user_question)

    if not results["documents"][0]:
        await cl.Message(content="Nessun candidato trovato.").send()
        return

    metadata = results["metadatas"][0][0]
    candidate_info = metadata.get("candidate_info", "Dati non disponibili")
    source_file = metadata.get("source", "File sconosciuto")
    context_text = results["documents"][0][0]


    prompt = f"""
    Sei un assistente HR. Utilizza i seguenti dettagli per rispondere alla domanda.
    
    DETTAGLI ANAGRAFICI: {candidate_info}
    FILE DI ORIGINE: {source_file}
    CONTENUTO CV: {context_text}
    
    DOMANDA UTENTE: {user_question}
    
    Istruzioni: Cita il nome del candidato e i suoi recapiti. Spiega perché è adatto basandoti sul contenuto del CV.
    """

    messages = cl.user_session.get("messages", [])
    messages.append({"role": "user", "content": prompt})

    response_message = cl.Message(content="")
    await response_message.send()

    try:
        # Chiamata unica 
        stream = LLMHelper.chat(messages)
        for chunk in stream:
            await response_message.stream_token(chunk)

        messages.append({"role": "assistant", "content": response_message.content})
        await response_message.update()
    except Exception as e:
        await cl.Message(content=f"Errore: {str(e)}").send()

    cl.user_session.set("messages", messages)