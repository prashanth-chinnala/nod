"""
The retrieval boundary. Imports nothing from the package, like `avatar.contracts`.

Deliberately absent from this interface: any notion of a turn, a session, an agent, or a
conversation. A retriever that knew about those would be making orchestration decisions,
which is the coupling this boundary exists to prevent — the same rule that keeps
`TalkingHeadRenderer` from knowing what a turn epoch is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Chunk:
    """
    One retrievable passage.

    `score` is comparable only within a single retriever's results. Keyword scores are term
    overlap and Chroma's are vector distances, so a threshold tuned against one is meaningless
    against the other — which is why nothing in the orchestration layer compares them to a
    constant.
    """

    text: str
    score: float
    document_id: str
    source: str = ""


@runtime_checkable
class Retriever(Protocol):
    """
    Turns a query into passages worth putting in front of the model.

    Implementations may block and may make a network call. The caller is responsible for
    deciding whether that cost is affordable inside a conversational turn — see the module
    docstring in `avatar.knowledge` for the measured figures that make it a real decision.
    """

    def index(self, document_id: str, text: str, *, source: str = "") -> int:
        """
        Add or replace a document. Returns the number of chunks it became.

        Replace rather than append, keyed on `document_id`: re-uploading an edited job
        description must not leave the previous version's chunks in the corpus, competing
        with the new ones and quietly retrieving stale requirements.
        """
        ...

    def retrieve(self, query: str, *, top_k: int = 3, budget_chars: int = 1200) -> list[Chunk]:
        """
        The best passages for `query`, most relevant first.

        `budget_chars` is a hard cap on the total returned, because this text lands in a
        system prompt on the critical path of every turn. Unbounded retrieval inflates
        time-to-first-token, which the latency budget shows is already the worst-behaved term
        in the pipeline.
        """
        ...

    def clear(self) -> None:
        """Drop everything indexed. Must be safe to call when empty."""
        ...
