"""
The real LLM: Claude, streamed, sentence-chunked.

Implements `SentenceStream`, so the orchestrator does not change and neither do its
tests. Everything it needs already existed: `chunk_into_sentences` turns the token
stream into speakable units, and the turn epoch cancels it.

**Verification status:** the request shape and the error path are verified against the
live API (auth succeeds, a malformed request is rejected as expected). A successful
completion has **never** been observed, because the account this was developed against
has no credit balance. Treat the first working turn as part of the work.

Three decisions worth defending, because each trades something real:

**`claude-opus-5`.** The strongest model, and the default. Time-to-first-token is what
matters for a conversation, not total generation time, and TTFT does not scale with
model size the way total generation does. If measured TTFT turns out to blow the budget
in §1.5, `claude-haiku-4-5` is the cheap swap -- but that is a decision to make against
a measurement, not in advance.

**Thinking disabled, effort low.** Anthropic's own guidance prefers thinking on at low
effort over disabling it, and for most applications that is right. It is wrong here:
thinking tokens are generated *before* any text, so they land directly on the one number
this whole system is built to minimise. The documented cost of disabling is that
internal `<thinking>` tags can occasionally leak into the response -- mitigated in the
system prompt below with the generically-phrased instruction the guidance recommends,
and *without* any "do not think" rule, which measurably makes leakage worse. The other
documented failure mode -- tool calls emitted as plain text -- cannot apply, because
this adapter declares no tools.

**No sampling parameters.** `temperature`, `top_p`, and `top_k` are rejected outright on
this model. Behaviour is steered by the prompt.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing

from avatar.contracts import Message
from avatar.llm import MAX_CHUNK_CHARS, chunk_into_sentences

DEFAULT_MODEL = "claude-opus-5"

MAX_TOKENS = 400
"""
A turn is one spoken question. At ~150wpm, 400 tokens is far more speech than anyone
wants to sit through, so this is a runaway guard rather than a budget -- the length
that actually matters is set by the system prompt.
"""

INTERVIEWER_SYSTEM = """\
You are conducting a live technical interview for a Head of Engineering role. Your \
words are spoken aloud to the candidate, so write for the ear, not the page.

Ask exactly one question per turn. Keep it under 40 words. No preamble, no numbered \
lists, no markdown, no headings, no emoji — none of that survives being read out.

Follow up on what the candidate actually said. If an answer was vague about a number, a \
trade-off, or a failure, ask for the specific. If they gave a good answer, go one level \
deeper rather than changing subject.

If the transcript shows your previous question was cut off mid-sentence, the candidate \
did not hear the end of it. Do not pretend they did.

Do not include internal or system XML tags in your response.\
"""
"""
The last line is not decoration.

Disabling thinking on this model can leak internal tags into the visible response.
Phrasing it generically -- rather than naming thinking tags -- is the mitigation the
model guidance recommends, and it is measurably more effective than naming them.
"""


class AnthropicInterviewer:
    """
    A `SentenceStream` backed by the Messages API.

    Cancellation is the interesting part. When the orchestrator abandons a turn it
    stops consuming this generator, and the generator's `finally` exits the streaming
    context manager, which aborts the HTTP request. Without that, a barge-in would
    leave the model generating -- and billing -- a response nobody will ever hear.
    `SessionOrchestrator._run_turn` closes the generator deterministically rather than
    leaving it to the garbage collector, which is what makes this reliable.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        system: str = INTERVIEWER_SYSTEM,
        max_tokens: int = MAX_TOKENS,
        effort: str = "low",
        max_chunk_chars: int = MAX_CHUNK_CHARS,
        client: object | None = None,
    ) -> None:
        self.model = model
        self.system = system
        self.max_tokens = max_tokens
        self.effort = effort
        self.max_chunk_chars = max_chunk_chars
        # Injected in tests so the suite never touches the network. In production the
        # SDK resolves credentials from the environment on its own.
        self._client = client if client is not None else _build_client()
        self.requests = 0
        self.last_usage: dict[str, int] = {}

    def __call__(self, history: Sequence[Message]) -> AsyncGenerator[str, None]:
        return chunk_into_sentences(self._tokens(history), max_chars=self.max_chunk_chars)

    async def _tokens(self, history: Sequence[Message]) -> AsyncGenerator[str, None]:
        """
        Raw token stream. Sentence chunking happens one layer up.

        The opening turn has no history, and the API requires the first message to be
        from the user -- so a session that has not heard anything yet gets a synthetic
        opener rather than an empty `messages` array.
        """
        messages: list[Message] = list(history) or [
            {"role": "user", "content": "[the candidate has joined and is ready to begin]"}
        ]
        self.requests += 1

        stream = self._client.messages.stream(  # type: ignore[attr-defined]
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            messages=messages,
            # Disabled deliberately -- see the module docstring. Accepted only at
            # effort `high` or below on this model; `low` is well inside that.
            thinking={"type": "disabled"},
            output_config={"effort": self.effort},
        )
        async with stream as live:
            # Third `aclosing` in the chain: chunker -> here -> the SDK's token
            # generator. `async for` closes nothing, so each wrapper has to propagate
            # the close explicitly or the one below it is left suspended. Exiting the
            # outer context manager is what actually aborts the HTTP request, but
            # closing the token generator first is what lets the SDK unwind cleanly
            # rather than being torn down mid-read.
            async with aclosing(live.text_stream) as tokens:
                async for token in tokens:
                    yield token
            final = await live.get_final_message()
            self.last_usage = {
                "input_tokens": final.usage.input_tokens,
                "output_tokens": final.usage.output_tokens,
            }
            if final.stop_reason == "refusal":
                # A safety classifier declined. Not an error and not retryable with the
                # same prompt; the turn simply produced nothing to say.
                self.last_usage["refused"] = 1


def _build_client() -> object:
    """
    Construct the async client, failing with a useful message rather than a stack trace.

    An unset key is the single most common way this is misconfigured, and the SDK's own
    error does not mention where the project expects the key to live.
    """
    try:
        from anthropic import AsyncAnthropic
    except ModuleNotFoundError as exc:  # pragma: no cover - environment, not logic
        raise RuntimeError("the Anthropic LLM needs the SDK: pip install -e '.[llm]'") from exc

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Put it in .env (gitignored) and run with "
            "`set -a && . ./.env && set +a`, or export it. "
            "Run with AVATAR_LLM=scripted to use the canned interviewer instead."
        )
    return AsyncAnthropic()


def build_llm(name: str = "scripted") -> object:
    """
    The one-line LLM swap, mirroring `renderers.build` and `vad.build_vad`.

    Defaults to `scripted` so a clean clone runs with no key and no network.
    """
    key = name.lower()
    if key == "scripted":
        from avatar.llm import ScriptedInterviewer

        return ScriptedInterviewer()
    if key == "anthropic":
        return AnthropicInterviewer()
    raise ValueError(f"unknown LLM {name!r}; available: 'scripted', 'anthropic'")
