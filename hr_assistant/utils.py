from config import Config
from openai import OpenAI

client = OpenAI(base_url=Config.AI_API_URL, api_key=Config.AI_API_KEY)


class LLMHelper:

    @staticmethod
    def chat(messages):
        return client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=messages,
            stream=True,
        )

    @staticmethod
    async def get_candidate_name(header_text: str) -> str:
        """
        Estrae il nome del candidato dalle prime righe del CV (testo grezzo).
        Usa temperature=0 per massima determinismo e un system prompt stretto
        che forza il formato 'Nome Cognome' senza frasi aggiuntive.
        """
        response = client.chat.completions.create(
            model=Config.LLM_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Sei un estrattore di dati da curriculum vitae. "
                        "Rispondi SOLO con il nome e cognome del candidato, "
                        "senza punteggiatura né frasi aggiuntive. "
                        "Se non trovi un nome, rispondi esattamente: Candidato Sconosciuto"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Estrai il nome completo del candidato dall'intestazione di questo CV.\n"
                        "Rispondi SOLO con 'Nome Cognome'.\n\n"
                        f"TESTO HEADER CV:\n{header_text}"
                    ),
                },
            ],
        )

        name = response.choices[0].message.content.strip()

        
        if not name or len(name.split()) > 5 or len(name) > 60:
            return "Candidato Sconosciuto"

        return name

    @staticmethod
    async def get_db_stats(context: str) -> str:
        response = client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Descrivi in modo sintetico le statistiche del database dei frammenti indicizzati. "
                        "Includi la percentuale di frammenti per file.\n\n"
                        f"Statistiche: {context}"
                    ),
                }
            ],
        )
        return response.choices[0].message.content

    @staticmethod
    def create_prompt(context: str, question: str, candidate_name: str = None) -> str:
        name_section = candidate_name if candidate_name else "Non identificato"
        return f"""
Dato il seguente contesto:
[[[
{context}
]]]

CANDIDATO IDENTIFICATO:
{name_section}

DOMANDA UTENTE:
[[[ {question} ]]]

ISTRUZIONI:
- Spiega perché il candidato è il più adatto
- Usa evidenze dal contesto
- Alla fine crea sezione contatti (nome, email, telefono)
- Inserisci il nome del file CV SOLO alla fine
"""
