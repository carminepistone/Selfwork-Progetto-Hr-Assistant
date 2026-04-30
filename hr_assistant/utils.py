from config import Config
from openai import OpenAI

client = OpenAI(base_url=Config.AI_API_URL, api_key=Config.AI_API_KEY)

class LLMHelper:

    @staticmethod
    def chat(messages):
        return client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=messages,
            stream=True
        )

    @staticmethod
    async def get_candidate_name(context):
        response = client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[{
                "role": "user",
                "content": f"""Dato il seguente contesto individua il nome e cognome del candidato 
                e ritorna solo il nome e cognome. Curriculum: {context}"""
            }]
        )
        return response.choices[0].message.content

    @staticmethod
    async def get_db_stats(context):
        response = client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[{
                "role": "user",
                "content": f"""Descrivi in modo sintetico le statistiche del database dei frammenti indicizzati.
                Includi la percentuale di frammenti per file. Statistiche: {context}"""
            }]
        )
        return response.choices[0].message.content

    @staticmethod
    def create_prompt(context, question, candidate_name=None):
        return f"""
    Dato il seguente contesto:
    [[[
    {context}
    ]]]

    CANDIDATO IDENTIFICATO:
    {candidate_name}

    DOMANDA UTENTE:
    [[[ {question} ]]]

    ISTRUZIONI:
    - Spiega perché il candidato è il più adatto
    - Usa evidenze dal contesto
    - Alla fine crea sezione contatti (nome, email, telefono)
    - Inserisci il nome del file CV SOLO alla fine
    """