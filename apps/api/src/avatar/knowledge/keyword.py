"""
BM25 retrieval over paragraph chunks. No dependency, no network, no embedding model.

The default, and the reason is measured rather than aesthetic: Chroma Cloud's query round
trip is **408ms** against a turn budget already at 2.7-5.8s for a sub-second target. This
runs in well under a millisecond, in-process. On a corpus of a job description and a rubric,
term overlap finds the right paragraph; the extra recall semantic search buys does not pay
for a third of a second per turn.

BM25 rather than plain term counting, because two of its corrections matter even at this
scale. Rare terms outweigh common ones, so "idempotency" counts for more than "the". And
term frequency saturates, so a chunk that repeats one query word ten times does not beat a
chunk containing all of them once — which is exactly the failure a naive count produces on
documents with a repeated heading.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from avatar.knowledge.contracts import Chunk

K1 = 1.5
"""
Term-frequency saturation. The standard value.

Higher lets repetition keep paying; lower flattens it. 1.5 is the usual choice and there is
no corpus here to tune it against — inventing a bespoke value would be a fabricated
optimisation.
"""

B = 0.75
"""
Length normalisation. Standard.

At 0 a long chunk that mentions everything wins by sheer size; at 1 length is fully
discounted. Paragraph chunks vary enough in length here that this correction is doing real
work.
"""

WORD = re.compile(r"[a-z0-9]+")

# A list literal of 38 words is one unreadable line, and this list gets edited by hand while
# reasoning about which words carry retrieval signal -- so it is written to be read.
STOPWORDS = frozenset(
    """a an and are as at be but by for from has have i if in into is it its of or
    that the their there they this to was were what when where which who will with you
    your""".split()  # noqa: SIM905
)
"""
A deliberately tiny list, and it got smaller once a test exercised it.

Only words that carry no retrieval signal in any query. Aggressive stopword lists strip
terms that turn out to matter, which is not hypothetical here: **"on" was in this list until
a test showed it turned `on-call` into `call`.** In a corpus about running production
systems, "on-call" is one of the highest-signal terms there is, and losing half of it to a
generic stopword list is exactly the failure this docstring warned about while committing it.

"not", "no", and "own" are absent for the same reason — negation and ownership are load
bearing in an interview.
"""


def tokenise(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower()) if w not in STOPWORDS]


def chunk_text(text: str) -> list[str]:
    """
    Split on blank lines: paragraphs.

    Not sentences, and not a fixed token window. A paragraph is the unit a human wrote as one
    idea, so it is the unit most likely to answer a question on its own. Fixed windows cut
    mid-thought and retrieve halves of two unrelated points.
    """
    parts = [block.strip() for block in re.split(r"\n\s*\n", text)]
    return [p for p in parts if p]


@dataclass
class _Indexed:
    document_id: str
    source: str
    text: str
    terms: Counter[str]
    length: int


@dataclass
class KeywordRetriever:
    """In-process BM25. Cheap enough to re-score the whole corpus on every query."""

    chunks: list[_Indexed] = field(default_factory=list)

    def index(self, document_id: str, text: str, *, source: str = "") -> int:
        # Replace, never append: re-uploading an edited document must not leave the previous
        # version's chunks competing with the new ones and retrieving stale requirements.
        self.chunks = [c for c in self.chunks if c.document_id != document_id]
        added = 0
        for body in chunk_text(text):
            terms = Counter(tokenise(body))
            if not terms:
                continue  # a chunk of pure punctuation retrieves nothing and dilutes averages
            self.chunks.append(_Indexed(document_id, source, body, terms, sum(terms.values())))
            added += 1
        return added

    def retrieve(self, query: str, *, top_k: int = 3, budget_chars: int = 1200) -> list[Chunk]:
        query_terms = tokenise(query)
        if not query_terms or not self.chunks:
            return []

        total = len(self.chunks)
        avg_length = sum(c.length for c in self.chunks) / total

        # Inverse document frequency, computed per query rather than cached. At this corpus
        # size the cache would be a correctness risk (stale after an index) for no measurable
        # gain.
        idf: dict[str, float] = {}
        for term in set(query_terms):
            containing = sum(1 for c in self.chunks if term in c.terms)
            # The +0.5 smoothing keeps a term present in every chunk from going negative,
            # which would make a matching chunk score *worse* than a non-matching one.
            idf[term] = math.log(1 + (total - containing + 0.5) / (containing + 0.5))

        scored: list[Chunk] = []
        for candidate in self.chunks:
            score = 0.0
            for term in set(query_terms):
                frequency = candidate.terms.get(term, 0)
                if not frequency:
                    continue
                norm = K1 * (1 - B + B * candidate.length / avg_length)
                score += idf[term] * frequency * (K1 + 1) / (frequency + norm)
            if score > 0:
                scored.append(
                    Chunk(candidate.text, score, candidate.document_id, candidate.source)
                )

        scored.sort(key=lambda c: c.score, reverse=True)
        return _within_budget(scored, top_k, budget_chars)

    def clear(self) -> None:
        self.chunks = []


class NullRetriever:
    """
    Retrieves nothing, successfully.

    The default when no knowledge base is attached. A null implementation rather than an
    `Optional[Retriever]` threaded through the orchestrator: every call site would need a
    branch, and one of them would eventually forget it.
    """

    def index(self, document_id: str, text: str, *, source: str = "") -> int:
        return 0

    def retrieve(self, query: str, *, top_k: int = 3, budget_chars: int = 1200) -> list[Chunk]:
        return []

    def clear(self) -> None:
        return None


def _within_budget(scored: list[Chunk], top_k: int, budget_chars: int) -> list[Chunk]:
    """
    Take the best chunks until either count or character budget runs out.

    The character cap is the one that matters: this text goes into a system prompt on every
    turn, and time-to-first-token is already the worst-behaved term in the latency budget.
    A chunk that would breach the budget is skipped rather than truncated — half a paragraph
    reads as corruption to the model and can invert the meaning of the sentence it cuts.
    """
    kept: list[Chunk] = []
    used = 0
    for chunk in scored:
        if len(kept) >= top_k:
            break
        if used + len(chunk.text) > budget_chars:
            continue
        kept.append(chunk)
        used += len(chunk.text)
    return kept
