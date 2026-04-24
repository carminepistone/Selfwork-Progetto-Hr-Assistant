import ollama
from config import Config

class LLMHelper:

    @staticmethod
    def chat(messages):
        """Gestisce lo streaming da Ollama (Llama 3.1)"""
        stream = ollama.chat(
            model=Config.LLM_MODEL,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            yield chunk['message']['content']

    @staticmethod
    async def get_candidate_name(context):
        """Estrae il nome con un prompt 'zero-shot' secco"""
        response = ollama.chat(
            model=Config.LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Sei un estrattore di dati preciso. Rispondi SOLO con il nome e cognome richiesti, senza introduzioni o spiegazioni."
                },
                {
                    "role": "user",
                    "content": f"Trova nome e cognome all'inizio di questo CV: {context}",
                }
            ],
        )
        return response['message']['content'].strip()

    @staticmethod
    def create_prompt(context, question, candidate_name):
        """Prompt strutturato per migliorare il ragionamento di Llama 3.1"""
        return f"""
### ISTRUZIONI HR
Analizza la richiesta dell'utente e confrontala con il profilo del candidato estratto dal database.

### CONTESTO CANDIDATO
- **Nome Candidato:** {candidate_name}
- **Dettagli dal CV:** {context}

### RICHIESTA UTENTE
"{question}"

### REGOLE DI RISPOSTA
1. Conferma che il profilo di {candidate_name} è stato individuato nel file indicato nel contesto.
2. Argomenta PERCHÉ è adatto basandoti esclusivamente sul testo del CV fornito.
3. Se le competenze non corrispondono alla domanda, dichiaralo onestamente.
4. Non inventare esperienze non scritte nel testo.
5. Mantieni un tono professionale e sintetico.
"""