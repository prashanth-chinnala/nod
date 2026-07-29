"""
Knowledge bases: documents split into paragraph chunks, retrieved by keyword score.

**Why keyword retrieval and deliberately no embeddings.** A retrieval step lands *inside* a
conversational turn that already measures 2.7-5.8s end to end (PROCESS.md §3.4), and that
budget has nothing to spare. An embedding model costs either a network hop per query or a
second model resident beside the renderer on a 16GB T4; a vector store costs a service to
run and a schema to migrate. What it buys, over a knowledge base of a handful of short
documents an operator pasted in, is the ability to match a paraphrase — which BM25 term
overlap already handles acceptably at this corpus size, because the operator and the
candidate are both using the domain's vocabulary. So the scorer is ~30 lines below with no
dependency, no vector store, and no network call.

Where that breaks, stated so the boundary is a decision and not an oversight: a corpus in
the thousands of documents, or questions phrased in vocabulary genuinely disjoint from the
source text ("how do I get paid?" against a document that only says "remuneration"). At
that point this is replaced by an embedding index, not extended — `rank()` is the only
function that would change.

**Chunks live in their own top-level list**, not nested inside each document, because BM25
needs corpus-wide document frequency: a term's weight depends on how many chunks across the
whole base contain it, so the scorer scans one flat list. Nesting would make every query
flatten it first.

Handlers here are `def`, not `async def`, on purpose. `Store` does blocking file I/O, and
this router is mounted on the same event loop that streams video frames — a sync handler
runs in Starlette's threadpool, where a slow disk cannot stall the frame pump.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from avatar.store import NotFound, Store, new_id, store

COLLECTION = "knowledge"
ID_PREFIX = "kb"
DOCUMENT_ID_PREFIX = "doc"
CHUNK_ID_PREFIX = "chunk"

NAME_MAX_LENGTH = 120
"""Long enough for a real filename or title, short enough to render in a table cell."""

TOP_K_DEFAULT = 3
TOP_K_MAX = 20
"""
Outer wall on how many chunks a query may return.

Every returned chunk is text that would be pasted into a prompt, so an unbounded `top_k` is
an unbounded prompt: slower generation, and the relevant chunk buried under near-misses.
`0` is rejected rather than clamped — a query that can never return anything looks
configured rather than broken.
"""

BM25_K1 = 1.2
BM25_B = 0.75
"""
Standard BM25 saturation and length-normalisation constants.

`k1` caps how much a repeated term can help, so a chunk that says "latency" nine times does
not outrank the chunk that answers the question. `b` discounts long chunks, which otherwise
win by containing more of everything. Untuned, because tuning them against a corpus this
small would be fitting noise.
"""

_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n+")
_WORD = re.compile(r"[a-z0-9']+")


def get_store() -> Store:
    """
    Indirection so tests can bind a `Store(tmp_path)` via `dependency_overrides`.

    Without it the routes close over the process-wide default and a test run writes into
    whatever `AVATAR_DATA_DIR` points at — i.e. someone's real console data.
    """
    return store


StoreDep = Annotated[Store, Depends(get_store)]


# -- chunking and scoring ---------------------------------------------------


def chunk_paragraphs(text: str) -> list[str]:
    """
    Split on blank lines; a paragraph is the retrieval unit.

    Blank lines rather than a fixed character window because a paragraph is a unit the
    author already decided was one idea, and a window that cuts mid-sentence produces
    chunks that score well and read as gibberish when pasted into a prompt. Line endings
    are normalised first: a file pasted from Windows has `\\r\\n` and would otherwise
    arrive as one single unsplittable chunk.
    """
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    return [para.strip() for para in _PARAGRAPH_BREAK.split(normalised) if para.strip()]


def tokenise(text: str) -> list[str]:
    """
    Lowercase alphanumeric words, apostrophes kept inside them.

    Punctuation is dropped rather than split on, so `latency,` and `latency` are the same
    term — without that, a query never matches a term that happened to end a sentence.
    """
    return _WORD.findall(text.lower())


class Retrieved(BaseModel):
    """One chunk the query would pull, with the evidence for why."""

    text: str
    score: float
    document_id: str


def rank(chunks: Sequence[Mapping[str, Any]], query: str, top_k: int) -> list[Retrieved]:
    """
    BM25 over the base's chunks. Highest score first, zero-scoring chunks omitted.

    Two properties this has that a raw term-overlap count does not, and both matter for the
    console's retrieval tester being trustworthy:

      * **rare terms dominate.** A chunk full of "the" must not outrank the one chunk that
        mentions the distinctive term in the query, which is exactly what counting matches
        would do.
      * **a miss is visibly a miss.** A chunk sharing no term scores zero and is dropped
        rather than padding the response to `top_k`, so an operator can tell "retrieval
        found nothing" from "retrieval found the wrong thing" — the whole point of exposing
        this in the UI.

    Ties keep insertion order, which is document order, so the same query on the same base
    always returns the same list.
    """
    terms = tokenise(query)
    tokenised = [tokenise(str(chunk.get("text", ""))) for chunk in chunks]
    if not terms or not tokenised:
        return []

    lengths = [len(tokens) for tokens in tokenised]
    total_length = sum(lengths)
    # Guarded division: a base whose every chunk tokenised to nothing would otherwise
    # divide by zero here instead of simply matching nothing.
    average_length = total_length / len(tokenised) if total_length else 1.0

    frequency: Counter[str] = Counter()
    for tokens in tokenised:
        frequency.update(set(tokens))

    corpus_size = len(tokenised)
    hits: list[Retrieved] = []
    for chunk, tokens, length in zip(chunks, tokenised, lengths, strict=True):
        counts = Counter(tokens)
        score = 0.0
        for term in terms:
            term_frequency = counts.get(term, 0)
            if term_frequency == 0:
                continue
            document_frequency = frequency[term]
            # The `log(1 + ...)` form of IDF, not the textbook one: the classic version goes
            # negative for a term appearing in more than half the corpus, which with a
            # handful of chunks would let a common term *penalise* the chunk containing it.
            idf = math.log(
                1 + (corpus_size - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            saturation = (
                term_frequency
                * (BM25_K1 + 1)
                / (term_frequency + BM25_K1 * (1 - BM25_B + BM25_B * length / average_length))
            )
            score += idf * saturation
        if score <= 0.0:
            continue
        hits.append(
            Retrieved(
                text=str(chunk.get("text", "")),
                # Rounded at the boundary: a UI column showing 1.4589999999999999 reads as
                # a bug in the scorer to anyone looking at it.
                score=round(score, 4),
                document_id=str(chunk.get("document_id", "")),
            )
        )

    hits.sort(key=lambda hit: -hit.score)
    return hits[:top_k]


# -- request bodies ---------------------------------------------------------


def _require_text(value: str, field: str) -> str:
    """
    Reject whitespace-only input, which `min_length` alone accepts.

    `" "` is what a half-filled form submits, and a knowledge base named `" "` is
    unselectable in the console's picker — present in the list, impossible to point at.
    """
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must not be blank")
    return text


class KnowledgeCreate(BaseModel):
    """
    A new, empty knowledge base.

    `extra="forbid"` is what stops a client supplying `chunk_count` or `documents`. Those
    are derived from the stored chunks on every write; accepting them would let the counters
    in the console disagree with the corpus that answers queries, and the list view would be
    confidently wrong with nothing to point at.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=NAME_MAX_LENGTH)
    description: str = ""

    @field_validator("name")
    @classmethod
    def _name_present(cls, value: str) -> str:
        return _require_text(value, "name")


class KnowledgePatch(BaseModel):
    """
    A partial update; omitted fields are left alone.

    Only the operator-owned fields are patchable. `Store.update` drops `None`, so a stray
    null in a form submission is a no-op rather than a field cleared.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=NAME_MAX_LENGTH)
    description: str | None = None

    @field_validator("name")
    @classmethod
    def _name_present_if_given(cls, value: str | None) -> str | None:
        return None if value is None else _require_text(value, "name")


class DocumentCreate(BaseModel):
    """A document to chunk into the base."""

    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1, max_length=NAME_MAX_LENGTH)
    text: str = Field(min_length=1)

    @field_validator("filename")
    @classmethod
    def _filename_present(cls, value: str) -> str:
        return _require_text(value, "filename")

    @field_validator("text")
    @classmethod
    def _text_yields_a_chunk(cls, value: str) -> str:
        """
        Text that chunks to nothing is rejected at the boundary.

        A document of blank lines would otherwise be stored, appear in the document list,
        contribute no chunk, and be permanently unretrievable — an operator would see the
        upload succeed and conclude retrieval was broken.
        """
        if not chunk_paragraphs(value):
            raise ValueError("text contains no non-blank paragraph to chunk")
        return value


class QueryRequest(BaseModel):
    """A retrieval probe against one base."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    top_k: int = Field(default=TOP_K_DEFAULT, ge=1, le=TOP_K_MAX)

    @field_validator("query")
    @classmethod
    def _query_present(cls, value: str) -> str:
        return _require_text(value, "query")


# -- routes -----------------------------------------------------------------

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


def _load(data: Store, kb_id: str) -> dict[str, Any]:
    try:
        return data.get(COLLECTION, kb_id)
    except NotFound:
        # 404 rather than a bare KeyError traceback: an unknown id is a routine client
        # mistake (stale tab, deleted record), not a server fault.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"no knowledge base {kb_id!r}"
        ) from None


@router.get("")
def list_knowledge(data: StoreDep) -> list[dict[str, Any]]:
    """Newest first, per `Store.list`. The console's table renders this order verbatim."""
    return data.list(COLLECTION)


@router.post("", status_code=status.HTTP_201_CREATED)
def create_knowledge(body: KnowledgeCreate, data: StoreDep) -> dict[str, Any]:
    """
    Create the base with its derived fields already present and zeroed.

    A record without `documents` would make every consumer — the console table, the query
    route — write its own "missing means empty" fallback, and the one that forgets crashes
    on a base nobody has uploaded to yet.
    """
    record = body.model_dump()
    record.update(documents=[], chunks=[], chunk_count=0, total_chars=0)
    return data.create(COLLECTION, ID_PREFIX, record)


@router.get("/{kb_id}")
def get_knowledge(kb_id: str, data: StoreDep) -> dict[str, Any]:
    return _load(data, kb_id)


@router.patch("/{kb_id}")
def update_knowledge(kb_id: str, body: KnowledgePatch, data: StoreDep) -> dict[str, Any]:
    _load(data, kb_id)
    return data.update(
        COLLECTION, kb_id, body.model_dump(exclude_unset=True, exclude_none=True)
    )


@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_knowledge(kb_id: str, data: StoreDep) -> None:
    try:
        data.delete(COLLECTION, kb_id)
    except NotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"no knowledge base {kb_id!r}"
        ) from None


@router.post("/{kb_id}/documents", status_code=status.HTTP_201_CREATED)
def add_document(kb_id: str, body: DocumentCreate, data: StoreDep) -> dict[str, Any]:
    """
    Chunk a document into the base and return the whole updated base.

    The counters are **recomputed from the full chunk list**, never incremented. An
    increment drifts the moment a write is retried or a record is edited by hand on disk,
    and a `chunk_count` that disagrees with the chunks is a lie the console has no way to
    detect. Recomputing is O(chunks) on a corpus of dozens, which is free.

    Returns the base rather than the document so the caller does not need a second GET to
    render the new totals — the console updates a row from this response.
    """
    record = _load(data, kb_id)
    document_id = new_id(DOCUMENT_ID_PREFIX)
    paragraphs = chunk_paragraphs(body.text)

    documents = list(record.get("documents") or [])
    documents.append(
        {
            "id": document_id,
            "filename": body.filename,
            "text": body.text,
            "chunk_count": len(paragraphs),
        }
    )

    chunks = list(record.get("chunks") or [])
    chunks.extend(
        {"id": new_id(CHUNK_ID_PREFIX), "document_id": document_id, "text": paragraph}
        for paragraph in paragraphs
    )

    return data.update(
        COLLECTION,
        kb_id,
        {
            "documents": documents,
            "chunks": chunks,
            "chunk_count": len(chunks),
            # Counted over chunks, not over the uploaded text: whitespace stripped during
            # chunking is never searched, so charging the base for it would overstate the
            # corpus an operator is reasoning about.
            "total_chars": sum(len(str(chunk.get("text", ""))) for chunk in chunks),
        },
    )


@router.post("/{kb_id}/query")
def query_knowledge(kb_id: str, body: QueryRequest, data: StoreDep) -> list[Retrieved]:
    """
    What this base would retrieve for a query, scored.

    Exposed as a route because a knowledge base is otherwise a black box: when an answer is
    wrong, this is the only thing that distinguishes bad retrieval from a bad generation,
    and without it every such bug is debugged by guessing at the prompt.
    """
    record = _load(data, kb_id)
    return rank(record.get("chunks") or [], body.query, body.top_k)


@router.post("/{kb_id}/embed")
def embed_knowledge(kb_id: str, data: StoreDep) -> dict[str, Any]:
    """
    Push this base's documents into Chroma Cloud, so semantic retrieval can use them.

    **Explicitly an action rather than a side effect of upload.** Embedding is not free and it
    is not always wanted: the measured cost of a Chroma query is **408ms**, against a turn that
    already runs 2.7-5.8s for a sub-second target, and the first call downloads a ~79MB ONNX
    model because Chroma embeds client-side. Uploading a document should not silently commit an
    operator to that. So it is a button, and the response reports what it cost.

    What this buys, and it is real: matching a paraphrase that shares no words with the
    document. Keyword scoring cannot do that, and on a large or varied corpus it is the
    difference between retrieving the right paragraph and retrieving nothing. On one job
    description it is not worth 408ms a turn, which is why the default retriever stays
    keyword-based even after this has run.

    Failure is reported rather than raised as a 500. A missing credential or an unreachable
    vector store is a configuration problem the operator can fix from the same screen, and a
    stack trace in a browser console is a worse way to learn it.
    """
    record = _load(data, kb_id)
    documents = [
        document
        for document in (record.get("documents") or [])
        if str(document.get("text") or "").strip()
    ]
    if not documents:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="nothing to embed: upload a document first",
        )

    from avatar.knowledge.chroma import ChromaRetriever

    started = time.perf_counter()
    try:
        retriever = ChromaRetriever(collection=f"nod_{kb_id}")
        chunks = 0
        for document in documents:
            chunks += retriever.index(
                f"{kb_id}:{document.get('id', '')}",
                str(document["text"]),
                source=str(document.get("filename") or ""),
            )
    # Broad on purpose: a missing credential, an unreachable store, and a client-library
    # change are all the same thing to an operator -- something to fix on this screen -- and a
    # stack trace in a browser console is a worse way to learn any of them.
    except Exception as exc:
        return {
            "embedded": False,
            "detail": f"{type(exc).__name__}: {exc}",
            "collection": f"nod_{kb_id}",
        }

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    return {
        "embedded": True,
        "collection": f"nod_{kb_id}",
        "documents": len(documents),
        "chunks": chunks,
        "elapsed_ms": elapsed_ms,
        # Returned so the console can say what switching to semantic retrieval would cost per
        # turn, rather than presenting it as free.
        "retriever_env": "AVATAR_RETRIEVER=chroma",
        "measured_query_ms": 408,
    }
