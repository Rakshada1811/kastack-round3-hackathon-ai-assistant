from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


RetrievalPath = Literal["semantic", "keyword", "hybrid"]


class UploadRequest(BaseModel):
    csv_path: str = "conversations.csv"
    max_conversations: int | None = Field(default=200, ge=1)
    rebuild: bool = True


class UploadResponse(BaseModel):
    conversations_indexed: int
    documents_indexed: int
    events_indexed: int
    semantic_backend: str
    keyword_backend: str


class AskRequest(BaseModel):
    question: str = Field(min_length=2)
    user_id: str = "demo-user"
    path_override: RetrievalPath | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    id: str
    score: float
    memory_type: str
    retrieval_path: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    answer: str
    route: RetrievalPath
    route_reason: str
    sources: list[Source]
