from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from app.ingestion import MemoryDocument


TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")


@dataclass(frozen=True)
class SearchHit:
    document: MemoryDocument
    score: float
    retrieval_path: str


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def merge_hits(*hit_lists: list[SearchHit], top_k: int = 5) -> list[SearchHit]:
    merged: dict[str, SearchHit] = {}
    score_parts: defaultdict[str, float] = defaultdict(float)
    best_path: dict[str, str] = {}

    for hits in hit_lists:
        for rank, hit in enumerate(hits):
            score_parts[hit.document.id] += hit.score + 1.0 / (rank + 1)
            if hit.document.id not in merged or hit.score > merged[hit.document.id].score:
                merged[hit.document.id] = hit
                best_path[hit.document.id] = hit.retrieval_path

    reranked = [
        SearchHit(
            document=hit.document,
            score=round(score_parts[doc_id], 6),
            retrieval_path=best_path[doc_id],
        )
        for doc_id, hit in merged.items()
    ]
    reranked.sort(key=lambda hit: hit.score, reverse=True)
    return reranked[:top_k]
