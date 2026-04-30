import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import chainlit as cl
from document_processor import DocumentProcessor
from database import Database
from config import Config
from utils import LLMHelper



# INIZIALIZZAZIONE DB  SYNC DOCUMENTI 
db = Database()
added, updated, removed = DocumentProcessor.process_documents(db)
print(f"Document sync complete: {added} added, {updated} updated, {removed} removed")


#  ACTION CALLBACK

@cl.action_callback("db_stats")
async def on_db_stats(action: cl.Action):
    db_info = db.get_stats()
    response = await LLMHelper.get_db_stats(db_info)
    await cl.Message(response).send()


@cl.action_callback("db_reindex")
async def on_db_reindex(action: cl.Action):
    added, updated, removed = DocumentProcessor.process_documents(db)
    await cl.Message(
        f"DB reindicizzato. Sync: {added} aggiunti, {updated} aggiornati, {removed} rimossi"
    ).send()


#  CHAT

@cl.on_chat_start
async def start():
    actions = [
        cl.Action(
            name="db_stats",
            icon="mouse-pointer-click",
            payload={"value": "db_stats"},
            label="Statistiche Database",
        ),
        cl.Action(
            name="db_reindex",
            icon="mouse-pointer-click",
            payload={"value": "db_reindex"},
            label="Reindex Database",
        ),
    ]
    await cl.Message(content="Informazioni del sistema:", actions=actions).send()

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


#   MESSAGGI 

@cl.on_message
async def handle_message(message: cl.Message):
    user_question = message.content

    results = db.query(user_question, 3)
    if not results["documents"][0]:
        await cl.Message(content="Nessun candidato trovato.").send()
        return

    metadata = results["metadatas"][0][0]
    filename = metadata.get("source", "File sconosciuto")

    candidate_info = DocumentProcessor.read_first_lines(
        os.path.join(Config.DOCUMENTS_DIR, filename), 10
    )

    context = (
        f"CONTESTO: nome file {filename}, "
        f"paragrafo più significativo: {results['documents'][0][0]}, "
        f"informazioni candidato: {candidate_info}"
    )

    prompt = LLMHelper.create_prompt(context, user_question)

    messages = cl.user_session.get("messages", [])
    messages.append({"role": "user", "content": prompt})

    response_message = cl.Message(content="")
    await response_message.send()

    try:
        stream = LLMHelper.chat(messages)
        for chunk in stream:
            await response_message.stream_token(
                str(chunk.choices[0].delta.content or "")
            )
        messages.append({"role": "assistant", "content": response_message.content})
        await response_message.update()
    except Exception as e:
        error_message = f"Errore: {str(e)}"
        await cl.Message(content=error_message).send()
        print(error_message)

    cl.user_session.set("messages", messages)