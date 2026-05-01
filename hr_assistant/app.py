import os
import shutil
import chainlit as cl
from typing import List
from document_processor import DocumentProcessor
from database import Database
from config import Config
from utils import LLMHelper



# Inizializzazione

db = Database()
dp = DocumentProcessor()

added, updated, removed = dp.process_documents(db)
print(f"Document sync complete: {added} added, {updated} updated, {removed} removed")


# UI Starter

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Ricerca candidato",
            message="Cercami un candidato con competenze Python e AI",
            icon="/public/idea.svg",
        )
    ]

# Actions

@cl.action_callback("db_stats")
async def on_db_stats(action: cl.Action):
    db_info = db.get_stats()
    response = await LLMHelper.get_db_stats(db_info)

    await cl.Message(
        content=response,
        author="assistant",
        actions=[
            cl.Action(
                name="db_stats",
                label="Aggiorna statistiche",
                payload={"value": "db_stats"},
            )
        ],
    ).send()


@cl.action_callback("db_reindex")
async def on_db_reindex(action: cl.Action):
    added, updated, removed = dp.process_documents(db)

    await cl.Message(
        content=f"Reindex completato: {added} aggiunti, {updated} aggiornati, {removed} rimossi",
        author="system",
    ).send()


@cl.action_callback("db_remove")
async def on_db_remove(action: cl.Action):
    db.delete_collection()
    await cl.Message(
        content="Database completamente rimosso. Eseguire reindex.",
        author="system",
    ).send()


# Chat

@cl.on_chat_start
async def start():

    actions = [
        cl.Action(
            name="db_stats",
            label="Statistiche DB",
            payload={"value": "db_stats"},
        ),
        cl.Action(
            name="db_reindex",
            label="Reindex DB",
            payload={"value": "db_reindex"},
        ),
        cl.Action(
            name="db_remove",
            label="Reset DB",
            payload={"value": "db_remove"},
        ),
    ]

    await cl.Message(content="Sistema pronto", author="system", actions=actions).send()

    cl.user_session.set(
        "messages",
        [
            {
                "role": "system",
                "content": """
Sei un assistente HR tecnico.
Analizza CV rispetto alle richieste.
Output:
- punti di forza
- gap
- fit complessivo
Risposte concise.
""",
            }
        ],
    )


# File processing

async def _process_and_index_file(file_path: str, file_name: str) -> str:
    documents, metadatas, ids = dp.process_single_document(file_path)

    if documents:
        db.add_documents(documents, metadatas, ids)
        return f"{file_name} indicizzato"
    return f"Errore processamento {file_name}"


async def _handle_upload(files) -> List[str]:
    results = []

    for file in files:
        dst = os.path.join(Config.DOCUMENTS_DIR, file.name)
        os.makedirs(Config.DOCUMENTS_DIR, exist_ok=True)
        shutil.move(file.path, dst)

        result = await _process_and_index_file(dst, file.name)
        results.append(result)

    return results


# Messaggi

@cl.on_message
async def handle_message(message: cl.Message):

    # Upload
    if message.elements:
        await cl.Message(
            content="Indicizzazione documenti...",
            author="system",         # stato operativo
        ).send()

        if valid_files:
            results = await _handle_upload(valid_files)
            await cl.Message(
                content="\n".join(results),
                author="system",
            ).send()
            await cl.Message(
                content=f"{len(valid_files)} file caricati",
                author="system",
            ).send()
        else:
            await cl.Message(
                content="Nessun file valido",
                author="system",
            ).send()
        return


    # Query
    user_question = message.content
    results = db.query(user_question, 3)

    if not results or not results.get("documents") or not results["documents"][0]:
        await cl.Message(
            content="Nessun risultato",
            author="system",
        ).send()
        return

    try:
        metadata = results["metadatas"][0][0]
        filename = metadata.get("source")
    except Exception as e:
        await cl.Message(content=f"Errore retrieval: {str(e)}", author="system").send()
        return

    # Context
    file_path = os.path.join(Config.DOCUMENTS_DIR, filename)

    context_lines = DocumentProcessor.read_first_lines(file_path, 100)

    context = (
        f"File: {filename}\n"
        f"Chunk: {results['documents'][0][0]}"
    )


    candidate_name = await LLMHelper.get_candidate_name(context_lines)

    prompt = LLMHelper.create_prompt(
        context,
        user_question,
        candidate_name
    )

    messages = cl.user_session.get("messages", [])
    messages.append({"role": "user", "content": prompt})

    response_message = cl.Message(content="", author="assistant")
    await response_message.send()

    #LLM streaming 
    try:
        stream = LLMHelper.chat(messages)

        for chunk in stream:
            await response_message.stream_token(
                str(chunk.choices[0].delta.content or "")
            )

        messages.append({"role": "assistant", "content": response_message.content})
        await response_message.update()

    except Exception as e:
        await cl.Message(content=f"Errore LLM: {str(e)}", author="system").send()

    cl.user_session.set("messages", messages)