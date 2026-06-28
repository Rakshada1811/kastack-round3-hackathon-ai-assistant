from __future__ import annotations

import logging

from app.ingestion import MemoryDocument, build_retrieval_documents, load_conversations
from app.retrieval.keyword import KeywordIndex
from app.retrieval.router import choose_retrieval_path
from app.retrieval.scoring import SearchHit, merge_hits
from app.retrieval.semantic import SemanticIndex
from app.schemas import AskResponse, Source


logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(self) -> None:
        self.semantic = SemanticIndex()
        self.keyword = KeywordIndex()
        self.documents: list[MemoryDocument] = []
        self.events: list[MemoryDocument] = []
        self.indexed = False

    def ingest_csv(self, csv_path: str, max_conversations: int | None = None, rebuild: bool = True) -> dict[str, int | str]:
        if self.indexed and not rebuild:
            return self.stats()

        conversations = load_conversations(csv_path, max_conversations)
        documents, events = build_retrieval_documents(conversations)
        all_docs = documents + events

        self.documents = documents
        self.events = events
        self.semantic.build(all_docs)
        self.keyword.build(all_docs)
        self.indexed = True
        return self.stats()

    def stats(self) -> dict[str, int | str]:
        return {
            "conversations_indexed": len(self.documents),
            "documents_indexed": len(self.documents) + len(self.events),
            "events_indexed": len(self.events),
            "semantic_backend": self.semantic.backend_name,
            "keyword_backend": self.keyword.backend_name,
        }

    def ask(self, question: str, path_override: str | None = None, top_k: int = 5) -> AskResponse:
        if not self.indexed:
            self.ingest_csv("conversations.csv", max_conversations=200)

        route, reason = choose_retrieval_path(question, path_override)  # type: ignore[arg-type]
        hits = self._retrieve(question, route, top_k)
        logger.info(
            "rag_route_selected",
            extra={
                "route": route,
                "reason": reason,
                "top_k": top_k,
                "source_count": len(hits),
                "semantic_backend": self.semantic.backend_name,
                "keyword_backend": self.keyword.backend_name,
            },
        )
        answer = self._compose_answer(question, hits)
        return AskResponse(
            answer=answer,
            route=route,
            route_reason=reason,
            sources=[
                Source(
                    id=hit.document.id,
                    score=hit.score,
                    memory_type=hit.document.memory_type,
                    retrieval_path=hit.retrieval_path,
                    text=self._trim(hit.document.text),
                    metadata=hit.document.metadata,
                )
                for hit in hits
            ],
        )

    def _retrieve(self, question: str, route: str, top_k: int) -> list[SearchHit]:
        if route == "semantic":
            return self.semantic.search(question, top_k)
        if route == "keyword":
            return self.keyword.search(question, top_k)
        return merge_hits(
            self.semantic.search(question, top_k=top_k),
            self.keyword.search(question, top_k=top_k),
            top_k=top_k,
        )

    def _compose_answer(self, question: str, hits: list[SearchHit]) -> str:
        if not hits:
            return "I could not find a matching memory for that question."

        evidence = " ".join(self._trim(hit.document.text, limit=260) for hit in hits[:3])
        return (
            "Based on the retrieved conversation memory, the most relevant evidence is: "
            f"{evidence}"
        )

    @staticmethod
    def _trim(text: str, limit: int = 500) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."
