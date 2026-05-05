import re
import numpy as np

from custom_embedding import CustomEmbeddingFunction
from sklearn.metrics.pairwise import cosine_similarity


class SemanticChunking:
    """
    Chunking semantico basato su embedding + cosine distance.
    """

    def __init__(self, breakpoint_percentile: int = 95, buffer_size: int = 1):
        self.embedding_function = CustomEmbeddingFunction()
        self.breakpoint_percentile = breakpoint_percentile
        self.buffer_size = buffer_size


    def _process_sentences(self, text: str):
        sentences = [
            {"sentence": s.strip(), "index": i}
            for i, s in enumerate(re.split(r"(?<=[.?!])\s+", text))
            if s.strip()
        ]

        for i, current in enumerate(sentences):
            start = max(0, i - self.buffer_size)
            end = min(len(sentences), i + self.buffer_size + 1)
            current["combined_sentence"] = " ".join(
                sentences[j]["sentence"] for j in range(start, end)
            )

        return sentences

    def _calculate_distances(self, sentences):
        if len(sentences) < 2:
            return []

        embeddings = self.embedding_function(
            [s["combined_sentence"] for s in sentences]
        )

        distances = []
        for i in range(len(sentences) - 1):
            sim = cosine_similarity(
                [embeddings[i]],
                [embeddings[i + 1]],
            )[0][0]
            distances.append(1 - sim)

        return distances


    def chunk_text(self, text: str):
        sentences = self._process_sentences(text)

        if len(sentences) < 2:
            return [text]

        distances = self._calculate_distances(sentences)

        if not distances:
            return [text]

        threshold = np.percentile(distances, self.breakpoint_percentile)
        split_points = [i for i, d in enumerate(distances) if d > threshold]

        chunks = []
        start = 0

        for point in split_points:
            chunk = " ".join(s["sentence"] for s in sentences[start: point + 1])
            chunks.append(chunk)
            start = point + 1

        if start < len(sentences):
            chunks.append(" ".join(s["sentence"] for s in sentences[start:]))

        return chunks
