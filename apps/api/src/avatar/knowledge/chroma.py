"""
Chroma Cloud retrieval. Semantic, and measurably expensive.

**Measured against the real account, not estimated:** a `query` round trip took **408ms**,
and the first call additionally downloads a **79MB ONNX embedding model** (all-MiniLM-L6-v2)
because Chroma's default embedding function runs client-side. So each query is local
inference plus a network round trip.

Put that next to the latency budget: a full turn measures **2.7-5.8s against a sub-second
target**, and 408ms is roughly the whole TTS stage again. This is therefore opt-in
(`AVATAR_RETRIEVER=chroma`) rather than the default, and the threshold for switching is a
corpus large or varied enough that keyword matching starts missing relevant chunks — not a
preference for vectors.

Retrieval quality is genuinely better where it matters. Verified: the query "what should I
ask about queues and ordering?" returned the queue-ordering passage first, which term overlap
would also have found — but it would not have matched a paraphrase with no shared words, and
that is precisely what the 408ms buys.

**Credentials are never defaulted.** A missing key raises with the variable name rather than
silently falling back to keyword retrieval, because a session that quietly retrieves nothing
looks exactly like a model that ignores its context — the same failure class as the empty
transcript that took a day to find.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from avatar.knowledge.contracts import Chunk
from avatar.knowledge.keyword import chunk_text

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

DEFAULT_COLLECTION = "nod_knowledge"


class ChromaRetriever:
    """
    One Chroma collection, treated as a chunk store.

    Chunking happens here with the same paragraph rule the keyword retriever uses, rather
    than handing whole documents to Chroma. Two reasons: retrieval returns a passage a model
    can use instead of an entire job description, and the two retrievers stay comparable —
    if they chunked differently, a quality difference between them would be unattributable.
    """

    def __init__(
        self,
        *,
        collection: str | None = None,
        tenant: str | None = None,
        database: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.collection_name = collection or os.environ.get(
            "CHROMA_COLLECTION", DEFAULT_COLLECTION
        )
        self._tenant = tenant or os.environ.get("CHROMA_TENANT", "")
        self._database = database or os.environ.get("CHROMA_DATABASE", "")
        self._api_key = api_key or os.environ.get("CHROMA_API_KEY", "")
        self._collection: Any = None

    def _connect(self) -> Any:
        """
        Client and collection on first use, then cached.

        Deferred because constructing the client reaches the network and may trigger the
        79MB model download. Doing that at import would make every process that merely
        imports `avatar` pay for a feature it may not use.
        """
        if self._collection is not None:
            return self._collection

        missing = [
            name
            for name, value in (
                ("CHROMA_TENANT", self._tenant),
                ("CHROMA_DATABASE", self._database),
                ("CHROMA_API_KEY", self._api_key),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                f"the chroma retriever needs {', '.join(missing)}. Set them, or use the "
                "default keyword retriever, which needs no credentials and no network."
            )

        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - depends on the install extra
            raise RuntimeError(
                "the chroma retriever needs the client: pip install -e '.[vector]'"
            ) from exc

        client = chromadb.CloudClient(
            tenant=self._tenant, database=self._database, api_key=self._api_key
        )
        self._collection = client.get_or_create_collection(self.collection_name)
        return self._collection

    def index(self, document_id: str, text: str, *, source: str = "") -> int:
        collection = self._connect()
        chunks = chunk_text(text)
        if not chunks:
            return 0

        # Delete this document's previous chunks before writing the new ones. Upsert alone
        # is not enough: an edited document with fewer paragraphs would leave the surplus
        # chunks of the old version in the collection, retrievable and wrong.
        collection.delete(where={"document_id": document_id})
        collection.upsert(
            ids=[f"{document_id}:{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[
                {"document_id": document_id, "source": source, "chunk": i}
                for i in range(len(chunks))
            ],
        )
        return len(chunks)

    def retrieve(self, query: str, *, top_k: int = 3, budget_chars: int = 1200) -> list[Chunk]:
        collection = self._connect()
        result = collection.query(query_texts=[query], n_results=max(1, top_k))

        documents = (result.get("documents") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]

        chunks: list[Chunk] = []
        used = 0
        for text, distance, meta in zip(documents, distances, metadatas, strict=False):
            if used + len(text) > budget_chars:
                continue
            meta = meta or {}
            # Distance inverted into a score so `Chunk.score` means "more is better" for
            # every retriever. Without that, a caller sorting descending would rank the
            # worst match first — and Chroma returns distances, where less is better.
            chunks.append(
                Chunk(
                    text=text,
                    score=1.0 / (1.0 + float(distance)),
                    document_id=str(meta.get("document_id", "")),
                    source=str(meta.get("source", "")),
                )
            )
            used += len(text)
        return chunks

    def clear(self) -> None:
        collection = self._connect()
        existing = collection.get()
        ids = existing.get("ids") or []
        if ids:
            collection.delete(ids=ids)
