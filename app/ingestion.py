from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


MESSAGE_RE = re.compile(r"(?m)^User\s*(?P<speaker>\d+):\s*(?P<text>.*?)(?=^User\s*\d+:|\Z)", re.S)
EVENT_PATTERNS = (
    re.compile(r"\b(?:I am|I'm|I work as|I study|studying|my job is)\b[^.!?\n]*[.!?]?", re.I),
    re.compile(r"\b(?:I like|I love|I enjoy|favorite|hobby|hobbies)\b[^.!?\n]*[.!?]?", re.I),
    re.compile(r"\b(?:I have|my dog|my cat|my family|my parents)\b[^.!?\n]*[.!?]?", re.I),
    re.compile(r"\b(?:moving to|moved to|live in|from)\b[^.!?\n]*[.!?]?", re.I),
    re.compile(r"\b(?:sad|happy|excited|stressed|anxious|miss|sorry)\b[^.!?\n]*[.!?]?", re.I),
)


@dataclass(frozen=True)
class MemoryDocument:
    id: str
    text: str
    memory_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_conversations(csv_path: str | Path, limit: int | None = None) -> list[str]:
    path = Path(csv_path)
    conversations: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or not row[0].strip():
                continue
            conversations.append(row[0].strip())
            if limit and len(conversations) >= limit:
                break
    return conversations


def parse_messages(conversation: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for idx, match in enumerate(MESSAGE_RE.finditer(conversation), start=1):
        messages.append(
            {
                "message_index": idx,
                "speaker": f"User {match.group('speaker')}",
                "text": " ".join(match.group("text").split()),
            }
        )
    return messages


def build_retrieval_documents(conversations: list[str]) -> tuple[list[MemoryDocument], list[MemoryDocument]]:
    documents: list[MemoryDocument] = []
    events: list[MemoryDocument] = []

    for conv_index, conversation in enumerate(conversations):
        messages = parse_messages(conversation)
        conversation_text = "\n".join(f"{m['speaker']}: {m['text']}" for m in messages)
        documents.append(
            MemoryDocument(
                id=f"conv-{conv_index}",
                text=conversation_text,
                memory_type="conversation",
                metadata={"conversation_index": conv_index, "message_count": len(messages)},
            )
        )

        for event_index, event_text in enumerate(extract_events(messages)):
            events.append(
                MemoryDocument(
                    id=f"event-{conv_index}-{event_index}",
                    text=event_text,
                    memory_type="event",
                    metadata={"conversation_index": conv_index},
                )
            )

    return documents, events


def extract_events(messages: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    events: list[str] = []
    for message in messages:
        text = message["text"]
        for pattern in EVENT_PATTERNS:
            for match in pattern.finditer(text):
                event = " ".join(match.group(0).strip(" .").split())
                if len(event) < 8 or event.lower() in seen:
                    continue
                seen.add(event.lower())
                events.append(f"{message['speaker']} mentioned: {event}.")
    return events
