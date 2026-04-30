import re
import numpy as np
from custom_embedding import CustomEmbeddingFunction
from sklearn.metrics.pairwise import cosine_similarity


class SemanticChunking:
    """
    Divide un testo in chunk semanticamente coerenti.
    Usa embedding (OpenAI, local o Ollama) per individuare
    i punti di separazione in base al significato del testo.
    """

    def __init__(self, breakpoint_percentile=95, buffer_size=1):
        self.embedding_function = CustomEmbeddingFunction()
        self.breakpoint_percentile = breakpoint_percentile
        self.buffer_size = buffer_size

    def _process_sentences(self, text):
        sentences = [
            {"sentence": s, "index": i}
            for i, s in enumerate(re.split(r"(?<=[.?!])\s+", text))
            if s.strip()  # ✅ MODIFICA: evita frasi vuote
        ]
        for i, current in enumerate(sentences):
            context_range = range(
                max(0, i - self.buffer_size),
                min(len(sentences), i + self.buffer_size + 1),
            )
            current["combined_sentence"] = " ".join(
                sentences[j]["sentence"] for j in context_range
            )
        return sentences

    def _calculate_distances(self, sentences):
        if len(sentences) < 2:
            return []  # ✅ MODIFICA: evita calcolo inutile e caso edge

        embeddings = self.embedding_function(
            [s["combined_sentence"] for s in sentences]
        )

        distances = []
        for i in range(len(sentences) - 1):
            distance = 1 - cosine_similarity(
                [embeddings[i]], [embeddings[i + 1]]
            )[0][0]
            distances.append(distance)

        return distances

    def chunk_text(self, text):
        sentences = self._process_sentences(text)

        # gestione testi troppo brevi
        if len(sentences) < 2:
            return [text]

        distances = self._calculate_distances(sentences)

        # gestione caso distances vuoto
        if len(distances) == 0:
            return [text]

        threshold = np.percentile(distances, self.breakpoint_percentile)

        split_points = [i for i, d in enumerate(distances) if d > threshold]

        chunks = []
        start = 0
        for point in split_points + [len(sentences) - 1]:
            chunk = " ".join(s["sentence"] for s in sentences[start: point + 1])
            chunks.append(chunk)
            start = point + 1

        return chunks