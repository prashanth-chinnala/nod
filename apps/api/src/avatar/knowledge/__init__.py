"""
Retrieval, behind a boundary, with two implementations and a measured reason to prefer one.

The interviewer asks generically strong questions. Given a job description and a rubric it
asks *role-specific* ones, which is the point of this package.

**The measurement that chose the default.** Chroma Cloud, exercised against the real
account: a `query` round trip measured **408ms**, including client-side embedding through a
79MB ONNX model that the client downloads on first use. Keyword retrieval over the same
corpus is sub-millisecond and local. A full turn already measures **2.7-5.8s against a
sub-second target** (`PROCESS.md` 1.5), so 408ms is not a rounding error — it is roughly the
entire TTS stage again, spent per turn, for better recall on a corpus of a few short
documents where keyword matching is competitive.

So `keyword` is the default and `chroma` is opt-in via `AVATAR_RETRIEVER=chroma`. That is a
threshold judgment rather than a permanent preference: semantic retrieval earns its 408ms
when the corpus is large enough or the phrasing varied enough that keyword matching starts
missing relevant chunks. It does not earn it on one job description.

**Why a Protocol rather than an if-statement.** Same reason as every other boundary here: the
retriever is swappable, the orchestration around it does not change, and a test can
substitute a fake without a network. `Retriever` imports nothing from this package.
"""

from __future__ import annotations

import os

from avatar.knowledge.contracts import Chunk, Retriever

__all__ = ["Chunk", "Retriever", "build_retriever"]

DEFAULT_RETRIEVER = "keyword"


def build_retriever(name: str | None = None, **options: object) -> Retriever:
    """
    Registry, mirroring `build_llm` and `build_vad`.

    Imports are inside each branch: `chroma` pulls in a client and, on first use, downloads
    an embedding model. Importing that at module scope would make `import avatar` slow and
    network-dependent for every caller, including the ones using keyword retrieval.
    """
    chosen = (name or os.environ.get("AVATAR_RETRIEVER") or DEFAULT_RETRIEVER).lower()

    if chosen in ("none", "null", ""):
        from avatar.knowledge.keyword import NullRetriever

        return NullRetriever()

    if chosen == "keyword":
        from avatar.knowledge.keyword import KeywordRetriever

        return KeywordRetriever(**options)  # type: ignore[arg-type]

    if chosen == "chroma":
        from avatar.knowledge.chroma import ChromaRetriever

        return ChromaRetriever(**options)  # type: ignore[arg-type]

    raise ValueError(
        f"unknown retriever {chosen!r}; available: 'keyword' (default, local, sub-ms), "
        "'chroma' (semantic, ~400ms per query), 'none'"
    )
