"""
Retrieval: chunking, BM25 scoring, and the budget that keeps it off the critical path.

No test here touches a network. The Chroma retriever's own behaviour is verified by a live
script rather than a unit test, because mocking a vector store proves only that the mock was
written to match the assumptions — and the assumption worth checking was its 408ms round
trip, which no mock can tell you.
"""

from __future__ import annotations

import pytest

from avatar.knowledge import Chunk, Retriever, build_retriever
from avatar.knowledge.keyword import (
    KeywordRetriever,
    NullRetriever,
    chunk_text,
    tokenise,
)

JOB_DESCRIPTION = """\
Head of Engineering, real-time infrastructure

You will own the latency budget for our conversational avatar pipeline, including
speech-to-text, language model selection, and text-to-speech.

Incident response and on-call ownership sit with this role. We expect you to have run
production systems where a regression was visible to customers within minutes.

Our ingest pipeline is queue-backed and historically assumed strict message ordering,
an assumption that has caused data corruption in the past.
"""

RUBRIC = """\
Score candidates on how concretely they describe trade-offs.

A strong answer names a specific number, a specific failure, or a specific decision they
would reverse.
"""


def loaded() -> KeywordRetriever:
    retriever = KeywordRetriever()
    retriever.index("jd", JOB_DESCRIPTION, source="job-description.md")
    retriever.index("rubric", RUBRIC, source="rubric.md")
    return retriever


# -- the boundary ----------------------------------------------------------


def test_both_implementations_satisfy_the_protocol() -> None:
    assert isinstance(KeywordRetriever(), Retriever)
    assert isinstance(NullRetriever(), Retriever)


def test_the_default_is_keyword_and_needs_no_credentials() -> None:
    """
    The measured reason: Chroma's query round trip is 408ms against a turn budget already
    at 2.7-5.8s for a sub-second target. A default that reaches the network would put that
    on every turn without anyone choosing it.
    """
    assert isinstance(build_retriever(), KeywordRetriever)


def test_an_unknown_retriever_names_the_alternatives_and_their_cost() -> None:
    with pytest.raises(ValueError, match=r"408ms|available"):
        build_retriever("pinecone")


def test_none_resolves_to_a_null_retriever() -> None:
    """The state when no knowledge base is attached, which must not be an error."""
    assert isinstance(build_retriever("none"), NullRetriever)


# -- chunking --------------------------------------------------------------


def test_chunks_split_on_blank_lines_not_sentences() -> None:
    """
    A paragraph is the unit a human wrote as one idea, so it is the unit most likely to
    answer a question alone. Sentence or fixed-window splitting retrieves halves of two
    unrelated points.
    """
    chunks = chunk_text(JOB_DESCRIPTION)

    assert len(chunks) == 4
    assert all(chunk.strip() == chunk for chunk in chunks)
    assert any("queue-backed" in chunk for chunk in chunks)


def test_empty_and_whitespace_blocks_are_dropped() -> None:
    assert chunk_text("one\n\n\n\n   \n\ntwo") == ["one", "two"]


def test_a_chunk_of_pure_punctuation_is_not_indexed() -> None:
    """It can never match a query, and it drags the average-length normalisation down."""
    retriever = KeywordRetriever()

    assert retriever.index("noise", "---\n\n***\n\nreal content here") == 1


def test_stopwords_are_dropped_but_the_list_stays_small() -> None:
    """
    Aggressive stopword lists remove terms that turn out to matter. In an interview corpus
    "own" and "not" carry real signal, so neither may be filtered.
    """
    assert tokenise("the ownership of on-call") == ["ownership", "on", "call"]
    assert "own" in tokenise("systems you own")
    assert "not" in tokenise("what would you not do")


# -- retrieval quality ----------------------------------------------------


def test_it_retrieves_the_paragraph_that_answers_the_query() -> None:
    hits = loaded().retrieve("what caused the data corruption?", top_k=1)

    assert len(hits) == 1
    assert "queue-backed" in hits[0].text


def test_rare_terms_outweigh_common_ones() -> None:
    """
    BM25's inverse document frequency, and the reason plain term counting is not enough:
    a query's distinctive word must decide the result, not its filler.
    """
    hits = loaded().retrieve("ordering", top_k=1)

    assert "message ordering" in hits[0].text


def test_a_query_matching_nothing_returns_nothing_rather_than_the_least_bad_chunk() -> None:
    """
    Returning a weak match would put irrelevant text into the system prompt on every turn,
    which is worse than returning none: the model treats it as context it must account for.
    """
    assert loaded().retrieve("photosynthesis in ferns") == []


def test_an_empty_query_retrieves_nothing() -> None:
    assert loaded().retrieve("") == []
    assert loaded().retrieve("the and of") == [], "stopwords alone are not a query"


def test_retrieval_from_an_empty_corpus_is_empty_not_an_error() -> None:
    assert KeywordRetriever().retrieve("anything") == []


def test_scores_are_descending() -> None:
    """Callers take the head of this list; unsorted output would silently pick the worst."""
    hits = loaded().retrieve("latency ownership incident", top_k=4)

    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


# -- re-indexing ----------------------------------------------------------


def test_reindexing_replaces_a_document_rather_than_appending() -> None:
    """
    The defect this prevents: an edited job description leaves its previous version's chunks
    in the corpus, competing with the new ones, and the interviewer asks about a requirement
    that was deleted.
    """
    retriever = KeywordRetriever()
    retriever.index("jd", "We require Kubernetes experience.")
    retriever.index("jd", "We require Postgres experience.")

    hits = retriever.retrieve("what experience is required?", top_k=5)

    joined = " ".join(h.text for h in hits)
    assert "Postgres" in joined
    assert "Kubernetes" not in joined, "the old version must be gone, not outranked"


def test_clear_empties_the_corpus_and_is_safe_when_already_empty() -> None:
    retriever = loaded()
    retriever.clear()
    retriever.clear()

    assert retriever.retrieve("latency") == []


# -- the budget ----------------------------------------------------------


def test_top_k_is_respected() -> None:
    assert len(loaded().retrieve("latency ownership incident ordering", top_k=2)) == 2


def test_the_character_budget_is_a_hard_cap() -> None:
    """
    This text lands in a system prompt on every turn, and time-to-first-token is already the
    worst-behaved term in the latency budget. Unbounded retrieval inflates it directly.
    """
    hits = loaded().retrieve("latency incident ordering ownership", top_k=5, budget_chars=200)

    assert sum(len(h.text) for h in hits) <= 200


def test_an_oversized_chunk_is_skipped_not_truncated() -> None:
    """
    Half a paragraph reads as corruption to the model, and a cut can invert the meaning of
    the sentence it lands in — "we do not require X" becoming "we do require X".
    """
    retriever = KeywordRetriever()
    retriever.index("long", "latency " * 400)
    retriever.index("short", "latency budget ownership")

    hits = retriever.retrieve("latency", budget_chars=100)

    assert all(len(h.text) <= 100 for h in hits)


def test_metadata_survives_retrieval() -> None:
    """Without the source, a retrieved claim cannot be traced back to the document it came
    from — which is the first question anyone asks about a surprising answer."""
    hits = loaded().retrieve("data corruption", top_k=1)

    assert hits[0].document_id == "jd"
    assert hits[0].source == "job-description.md"


# -- the null retriever ---------------------------------------------------


def test_the_null_retriever_accepts_indexing_and_retrieves_nothing() -> None:
    """
    A null object rather than an Optional threaded through the orchestrator: every call site
    would need a branch, and one of them would eventually forget it.
    """
    null = NullRetriever()

    assert null.index("jd", JOB_DESCRIPTION) == 0
    assert null.retrieve("anything") == []
    null.clear()


def test_chunk_is_immutable() -> None:
    """Retrieved context must not be mutable by whatever it is handed to."""
    chunk = Chunk("text", 1.0, "doc")

    with pytest.raises((AttributeError, TypeError)):
        chunk.score = 2.0  # type: ignore[misc]
