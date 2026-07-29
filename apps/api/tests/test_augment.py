"""
Retrieval and pronunciation as boundary decorators.

Both wrap callables the orchestrator already takes, so these tests need no session, no clock,
and no transport — which is the point of putting them there rather than in the state machine.

The pronunciation tests are mostly about *not* mangling text. A lexicon that corrupts ordinary
words gets switched off, and then it protects nothing, so each failure mode gets its own test.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence

from avatar.contracts import AudioChunk, Message
from avatar.knowledge.augment import (
    CONTEXT_HEADER,
    apply_lexicon,
    latest_candidate_text,
    with_knowledge,
    with_pronunciation,
)
from avatar.knowledge.contracts import Chunk


class RecordingLlm:
    """Captures the history it was handed, which is the thing under test."""

    def __init__(self) -> None:
        self.seen: list[Sequence[Message]] = []

    def __call__(self, history: Sequence[Message]) -> AsyncGenerator[str, None]:
        self.seen.append(list(history))

        async def stream() -> AsyncGenerator[str, None]:
            yield "A question."

        return stream()


class RecordingTts:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def __call__(self, text: str, epoch: int) -> AsyncGenerator[AudioChunk, None]:
        self.spoken.append(text)

        async def stream() -> AsyncGenerator[AudioChunk, None]:
            yield AudioChunk(pcm=b"\x00\x00", epoch=epoch, duration_ms=40)

        return stream()


class FixedRetriever:
    def __init__(self, *chunks: str) -> None:
        self.chunks = [Chunk(text, 1.0, "doc") for text in chunks]
        self.queries: list[str] = []

    def index(self, document_id: str, text: str, *, source: str = "") -> int:
        return 0

    def retrieve(self, query: str, *, top_k: int = 3, budget_chars: int = 1200) -> list[Chunk]:
        self.queries.append(query)
        return self.chunks[:top_k]

    def clear(self) -> None:
        self.chunks = []


async def drain(stream: AsyncGenerator[str, None]) -> list[str]:
    return [item async for item in stream]


# -- what retrieval keys on ------------------------------------------------


def test_the_query_is_the_candidates_latest_answer() -> None:
    """
    Not the whole conversation: a query built from every turn drifts toward whatever was
    discussed most, so by turn six it retrieves context for turn one.
    """
    history: list[Message] = [
        {"role": "user", "content": "first answer about caching"},
        {"role": "assistant", "content": "a question"},
        {"role": "user", "content": "second answer about queues"},
    ]

    assert latest_candidate_text(history) == "second answer about queues"


def test_the_assistants_own_words_are_never_the_query() -> None:
    """Retrieving against the interviewer's question is a feedback loop that reinforces
    whatever it already asked."""
    history: list[Message] = [{"role": "assistant", "content": "tell me about queues"}]

    assert latest_candidate_text(history) == ""


def test_an_empty_history_produces_no_query() -> None:
    assert latest_candidate_text([]) == ""


# -- injection -------------------------------------------------------------


async def test_retrieved_context_is_appended_as_a_system_message() -> None:
    """
    Appended, not prepended. A system message at the front competes with the interviewer's
    own instructions and can override its tone; one at the end reads as material for this
    turn, which is what it is.
    """
    llm, retriever = RecordingLlm(), FixedRetriever("The ingest pipeline assumes ordering.")
    wrapped = with_knowledge(llm, retriever)

    await drain(wrapped([{"role": "user", "content": "we had a corruption incident"}]))

    handed = llm.seen[0]
    assert handed[-1]["role"] == "system"
    assert CONTEXT_HEADER in str(handed[-1]["content"])
    assert "assumes ordering" in str(handed[-1]["content"])
    assert handed[0]["role"] == "user", "the original history must come first, unmodified"


async def test_retrieving_nothing_leaves_the_history_untouched() -> None:
    """
    The common case early in a conversation. An empty `Relevant context:` header would read
    to the model as "there is no relevant context" — a stronger and less true statement than
    saying nothing at all.
    """
    llm = RecordingLlm()
    wrapped = with_knowledge(llm, FixedRetriever())

    history: list[Message] = [{"role": "user", "content": "anything"}]
    await drain(wrapped(history))

    assert llm.seen[0] == history


async def test_no_candidate_turn_means_no_retrieval_call_at_all() -> None:
    """The first turn has no answer to retrieve against, and querying with an empty string
    would score every chunk equally and inject an arbitrary one."""
    retriever = FixedRetriever("something")
    wrapped = with_knowledge(RecordingLlm(), retriever)

    await drain(wrapped([]))

    assert retriever.queries == []


async def test_the_wrapped_stream_still_yields_the_models_sentences() -> None:
    """The decorator must be transparent to output; only the input is changed."""
    wrapped = with_knowledge(RecordingLlm(), FixedRetriever("context"))

    assert await drain(wrapped([{"role": "user", "content": "hi"}])) == ["A question."]


# -- the lexicon: correctness, and mostly not-mangling ---------------------


def test_a_term_is_respelled() -> None:
    assert apply_lexicon("we run nginx", [("nginx", "engine ex")]) == "we run engine ex"


def test_matching_is_case_insensitive_but_replacement_is_literal() -> None:
    """Operators type the term however they think of it; the respelling is what they wrote."""
    assert apply_lexicon("NGINX and Nginx", [("nginx", "engine ex")]) == "engine ex and engine ex"


def test_a_substring_is_not_matched() -> None:
    """
    The defect: mapping "Kafka" turns "Kafkaesque" into "KAFF-ka-esque". Word boundaries are
    on the pattern for exactly this.
    """
    assert apply_lexicon("Kafkaesque", [("Kafka", "KAFF-ka")]) == "Kafkaesque"
    assert apply_lexicon("Kafka topic", [("Kafka", "KAFF-ka")]) == "KAFF-ka topic"


def test_a_replacement_is_never_itself_re_substituted() -> None:
    """
    Applying entries sequentially lets one rewrite feed the next: map SQL to "sequel" and
    PostgreSQL to "post-gress-sequel", and the second's output contains "sequel", which the
    first would rewrite again into "post-gress-sequel" recursively.
    """
    entries = [("SQL", "sequel"), ("PostgreSQL", "post gress sequel")]

    assert apply_lexicon("PostgreSQL", entries) == "post gress sequel"


def test_the_longer_term_wins() -> None:
    """Otherwise "SQL" consumes part of "PostgreSQL" and the longer entry never fires."""
    entries = [("SQL", "sequel"), ("PostgreSQL", "post gress")]

    assert apply_lexicon("we use PostgreSQL here", entries) == "we use post gress here"


def test_an_empty_lexicon_returns_the_text_unchanged() -> None:
    assert apply_lexicon("untouched", []) == "untouched"


def test_a_blank_term_is_ignored_rather_than_matching_everywhere() -> None:
    """An empty pattern with word boundaries matches between every character, which would
    interleave the replacement through the entire sentence."""
    assert apply_lexicon("hello world", [("", "X"), ("  ", "Y")]) == "hello world"


def test_regex_metacharacters_in_a_term_are_literal() -> None:
    """Operators type product names, not patterns. "C++" or "." must not compile as regex."""
    assert apply_lexicon("we use C++ daily", [("C++", "see plus plus")]) == "we use see plus plus daily"


def test_punctuation_around_a_term_survives() -> None:
    assert apply_lexicon("Is it nginx?", [("nginx", "engine ex")]) == "Is it engine ex?"


# -- the pronunciation decorator ------------------------------------------


async def test_the_synthesiser_receives_respelled_text() -> None:
    tts = RecordingTts()
    wrapped = with_pronunciation(tts, [("nginx", "engine ex")])

    async for _ in wrapped("restart nginx now", 1):
        pass

    assert tts.spoken == ["restart engine ex now"]


def test_an_empty_lexicon_returns_the_synthesiser_itself() -> None:
    """No wrapper at all in the common case, so the default path adds not even a call."""
    tts = RecordingTts()

    assert with_pronunciation(tts, []) is tts


async def test_the_epoch_is_passed_through_unchanged() -> None:
    """The epoch is how a barge-in invalidates audio already in flight. A decorator that
    dropped or rewrote it would break cancellation while looking correct."""
    tts = RecordingTts()
    wrapped = with_pronunciation(tts, [("a", "b")])

    chunks = [chunk async for chunk in wrapped("a", 7)]

    assert [chunk.epoch for chunk in chunks] == [7]
