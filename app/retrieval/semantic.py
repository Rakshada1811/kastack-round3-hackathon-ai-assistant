from __future__ import annotations

import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.ingestion import MemoryDocument
from app.retrieval.qdrant_store import QdrantVectorStore
from app.retrieval.scoring import SearchHit


class SemanticIndex:
    """Semantic search with SentenceTransformers when cached, TF-IDF fallback for local demos."""

    def __init__(self) -> None:
        self.documents: list[MemoryDocument] = []
        self.model = None
        self.matrix = None
        self.vectorizer: TfidfVectorizer | None = None
        self.qdrant = QdrantVectorStore()
        self.backend_name = "tfidf-fallback"
        self._model_load_attempted = False

    def _try_load_sentence_transformer(self) -> None:
        if os.getenv("USE_SENTENCE_TRANSFORMERS", "0").lower() not in {"1", "true", "yes"}:
            return

        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        local_only = os.getenv("SENTENCE_TRANSFORMERS_LOCAL_ONLY", "0").lower() in {"1", "true", "yes"}
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name, local_files_only=local_only)
            self.backend_name = f"sentence-transformers:{model_name}"
        except Exception:
            self.model = None

    def build(self, documents: list[MemoryDocument]) -> None:
        if not self._model_load_attempted:
            self._try_load_sentence_transformer()
            self._model_load_attempted = True

        self.documents = documents
        texts = [doc.text for doc in documents]
        if self.model:
            self.matrix = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            if self.qdrant.enabled:
                self.qdrant.recreate_collection(vector_size=int(self.matrix.shape[1]))
                self.qdrant.upsert(documents, np.asarray(self.matrix))
                self.backend_name += "+qdrant"
            return

        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=20000)
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        if not self.documents or self.matrix is None:
            return []

        if self.model:
            query_vector = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
            qdrant_hits = self.qdrant.search(np.asarray(query_vector), top_k=top_k)
            if qdrant_hits:
                by_id = {doc.id: doc for doc in self.documents}
                return [
                    SearchHit(
                        document=by_id[payload["document_id"]],
                        score=round(float(hit["score"]), 6),
                        retrieval_path="semantic",
                    )
                    for hit in qdrant_hits
                    for payload in [hit["payload"]]
                    if payload.get("document_id") in by_id
                ]
            scores = np.asarray(query_vector @ self.matrix.T).ravel()
        else:
            assert self.vectorizer is not None
            query_vector = self.vectorizer.transform([query])
            scores = cosine_similarity(query_vector, self.matrix).ravel()

        best_indices = np.argsort(scores)[::-1][:top_k]
        return [
            SearchHit(
                document=self.documents[int(idx)],
                score=round(float(scores[int(idx)]), 6),
                retrieval_path="semantic",
            )
            for idx in best_indices
            if float(scores[int(idx)]) > 0
        ]
