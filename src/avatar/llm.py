"""
The LLM boundary: a token stream turned into speakable units.

Sentence chunking lives on this side of the `SentenceStream` Protocol because it is
what lets TTS start before generation finishes. That overlap is the single reason a
sub-second turnaround is achievable at all: with identical component performance,
running the same stages sequentially-and-complete produces a multi-second
turnaround. Everything downstream -- TTS, renderer, transport -- is already
streaming; if this stage buffers the whole response, none of that matters.

No real model is wired in yet. `ScriptedInterviewer` is a stand-in that exercises
the streaming path honestly: it yields more than one sentence, so the TTS really
does start on the first while the second is still arriving.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

from avatar.contracts import Message, Sleep

SENTENCE_TERMINATORS = frozenset(".?!")

MAX_CHUNK_CHARS = 180
"""
Force a flush after this many characters with no terminator in sight.

A model that produces a long run without punctuation -- a list, a code block, a
stretch of prose it never ends -- would otherwise buffer until it finished, which
is precisely the stall this module exists to prevent. Flushing mid-clause makes
the TTS prosody slightly worse and the latency dramatically better.
"""

DEFAULT_QUESTIONS: tuple[str, ...] = (
    "Tell me about a system you designed that failed. What broke, and what did you "
    "change afterwards?",
    "Walk me through how you would cut a live feature over from a third-party vendor. "
    "Assume you cannot take a customer-facing regression.",
    "Where does the latency budget go in a system you have worked on recently? "
    "Name the term you would attack first.",
    "Describe a time you were the one holding an unpopular technical position. "
    "How did it resolve?",
)


def split_sentences(text: str) -> list[str]:
    """
    Split on sentence terminators, keeping the terminator and trailing space.

    Naive by design: it will split "Dr. Smith" and "e.g." in the wrong place. The
    cost of that error is a slightly odd TTS pause, and the fix is a real sentence
    segmenter -- worth it in production, not worth a dependency here. Noted rather
    than silently accepted.
    """
    out: list[str] = []
    current: list[str] = []
    for char in text:
        current.append(char)
        if char in SENTENCE_TERMINATORS:
            out.append("".join(current))
            current = []
    if current:
        out.append("".join(current))
    return out


async def chunk_into_sentences(
    tokens: AsyncIterator[str], *, max_chars: int = MAX_CHUNK_CHARS
) -> AsyncIterator[str]:
    """
    Turn a token stream into speakable units, emitting as early as possible.

    This is the piece M4 keeps when a real model replaces the scripted one: the
    model adapter yields tokens, this yields sentences, and nothing downstream
    changes.
    """
    buffer: list[str] = []
    size = 0
    async for token in tokens:
        buffer.append(token)
        size += len(token)
        ends_sentence = bool(token) and token.rstrip()[-1:] in SENTENCE_TERMINATORS
        if ends_sentence or size >= max_chars:
            yield "".join(buffer)
            buffer.clear()
            size = 0
    if buffer:
        yield "".join(buffer)


class ScriptedInterviewer:
    """
    A `SentenceStream` that asks canned questions, one per turn.

    Exists so the session layer, transport, and client can be demonstrated
    end-to-end before an LLM is wired in. It reads `history` only to decide how far
    through the script it is, which is enough to make the truncation tests
    meaningful: if history is truncated, the question count is too.
    """

    def __init__(
        self,
        questions: Sequence[str] = DEFAULT_QUESTIONS,
        *,
        ttft_ms: int = 180,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if not questions:
            raise ValueError("need at least one question")
        self._questions = list(questions)
        self._ttft_ms = ttft_ms
        self._sleep = sleep

    def __call__(self, history: Sequence[Message]) -> AsyncIterator[str]:
        asked = sum(1 for m in history if m["role"] == "assistant")
        return self._generate(self._questions[asked % len(self._questions)])

    async def _generate(self, question: str) -> AsyncIterator[str]:
        # Stands in for time-to-first-token. Only TTFT matters for perceived
        # latency, not total generation time, which is why it is modelled here and
        # the inter-sentence gap is not.
        await self._sleep(self._ttft_ms / 1000)
        for sentence in split_sentences(question):
            yield sentence
