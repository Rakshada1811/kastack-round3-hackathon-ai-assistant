from __future__ import annotations

from math import log

from app.ingestion import MemoryDocument
from app.retrieval.scoring import SearchHit, tokenize


class KeywordIndex:
    """Small BM25-style keyword index with no external runtime dependency."""

    backend_name = "local-bm25"

    def __init__(self) -> None:
        self.documents: list[MemoryDocument] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_freq: dict[str, int] = {}
        self.avg_doc_len = 1.0

    def build(self, documents: list[MemoryDocument]) -> None:
        self.documents = documents
        self.doc_tokens = [tokenize(doc.text) for doc in documents]
        self.doc_freq = {}
        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
        self.avg_doc_len = sum(len(tokens) for tokens in self.doc_tokens) / max(1, len(self.doc_tokens))

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        query_tokens = tokenize(query)
        if not query_tokens or not self.documents:
            return []

        scores: list[tuple[int, float]] = []
        total_docs = len(self.documents)
        k1 = 1.5
        b = 0.75

        for idx, tokens in enumerate(self.doc_tokens):
            term_counts: dict[str, int] = {}
            for token in tokens:
                term_counts[token] = term_counts.get(token, 0) + 1

            score = 0.0
            doc_len = len(tokens)
            for token in query_tokens:
                freq = term_counts.get(token, 0)
                if freq == 0:
                    continue
                df = self.doc_freq.get(token, 0)
                idf = log(1 + (total_docs - df + 0.5) / (df + 0.5))
                denom = freq + k1 * (1 - b + b * doc_len / self.avg_doc_len)
                score += idf * (freq * (k1 + 1)) / denom

            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        return [
            SearchHit(document=self.documents[idx], score=round(score, 6), retrieval_path="keyword")
            for idx, score in scores[:top_k]
        ]
