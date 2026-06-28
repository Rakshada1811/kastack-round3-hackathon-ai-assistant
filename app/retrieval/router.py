from __future__ import annotations

import re

from app.schemas import RetrievalPath


KEYWORD_HINTS = re.compile(r'["\']|exact|keyword|mentioned|say|said|named|called|list|where|when', re.I)
SEMANTIC_HINTS = re.compile(
    r"\b(why|how|what does|what can|summari[sz]e|personality|interest|prefer|mood|feel|similar|recommend)\b",
    re.I,
)
AMBIGUOUS_HINTS = re.compile(r"\b(tell me about|about the user|remember|know about|anything about)\b", re.I)


def choose_retrieval_path(question: str, override: RetrievalPath | None = None) -> tuple[RetrievalPath, str]:
    if override:
        return override, f"Caller forced route to {override}."

    keyword = bool(KEYWORD_HINTS.search(question))
    semantic = bool(SEMANTIC_HINTS.search(question))
    ambiguous = bool(AMBIGUOUS_HINTS.search(question))

    if ambiguous or (keyword and semantic):
        return "hybrid", "Question has both exact-memory and open-ended intent, so semantic and keyword results are merged."
    if keyword:
        return "keyword", "Question appears to ask for exact mentions, names, lists, places, or time-specific facts."
    if semantic:
        return "semantic", "Question asks for inferred meaning, preference, mood, or summary-style context."
    return "hybrid", "No strong signal was detected, so hybrid retrieval is safest for recall."
