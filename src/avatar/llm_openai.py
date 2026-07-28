"""
The same interviewer, on OpenAI.

This module earns its place by being boring. It implements `SentenceStream`, the
orchestrator does not change, its tests do not change, and neither does the client --
`AVATAR_LLM=openai` is the entire switch. Two unrelated providers behind one Protocol is
stronger evidence for "the model is a bounded, swappable piece" than either alone, which
is why the Anthropic adapter stays rather than being replaced.

What is genuinely different between the two, and therefore what the boundary is actually
absorbing:

| | Anthropic | OpenAI |
|---|---|---|
| system prompt | top-level `system` argument | a `role: "system"` entry in `messages` |
| output cap | `max_tokens` | `max_completion_tokens` |
| sampling | rejected outright on Opus 5 | accepted |
| thinking | on by default; must be disabled for TTFT | no equivalent knob |
| stream shape | `stream.text_stream` yields text | chunks carrying `choices[0].delta.content` |
| close | exiting the context manager aborts | same |

None of that reaches the state machine. It all lands here.

**Verification status:** the key authenticates and the request reaches the API, which
rejects it with `429 insufficient_quota` -- a billing gate, not a shape error. A
successful completion has **never** been observed on either provider, for the same
reason: no credit on either account. Treat the first working turn as part of the work.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing

from avatar.contracts import Message
from avatar.llm import MAX_CHUNK_CHARS, chunk_into_sentences

DEFAULT_MODEL = "gpt-4o"
"""
The stronger general model, not the cheaper one.

`gpt-4o-mini` has measurably lower time-to-first-token, and TTFT is the number this whole
system exists to minimise -- so it is a real candidate. It is not the default because
choosing it now would be picking a smaller model on a *guess* about a latency budget that
has not been measured yet. `AVATAR_LLM_MODEL=gpt-4o-mini` makes the comparison a one-line
change, and PROCESS.md 1.5 is where the answer belongs once there is one.
"""

MAX_COMPLETION_TOKENS = 400
"""
A runaway guard, not a budget.

One spoken interview question is far shorter than this; the length that matters is set by
the system prompt. Note the parameter name: `max_tokens` is deprecated on current models.
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
did not hear the end of it. Do not pretend they did.\
"""
"""
Deliberately *not* carrying the Anthropic adapter's trailing instruction about internal
XML tags. That line mitigates a documented failure mode of disabling thinking on Claude,
and has no counterpart here. Copying provider-specific mitigations between adapters is
how prompts accumulate cargo.
"""


class OpenAIInterviewer:
    """
    A `SentenceStream` backed by Chat Completions.

    Cancellation works the same way as the Anthropic adapter, and for the same reason:
    when the orchestrator abandons a turn it closes this generator, the `finally` unwinds
    the streaming context manager, and that aborts the HTTP request. Without it a
    barge-in leaves the provider generating -- and billing -- a response nobody hears.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        system: str = INTERVIEWER_SYSTEM,
        max_completion_tokens: int = MAX_COMPLETION_TOKENS,
        max_chunk_chars: int = MAX_CHUNK_CHARS,
        client: object | None = None,
    ) -> None:
        self.model = model or os.environ.get("AVATAR_LLM_MODEL", DEFAULT_MODEL)
        self.system = system
        self.max_completion_tokens = max_completion_tokens
        self.max_chunk_chars = max_chunk_chars
        # Injected in tests so the suite never touches the network.
        self._client = client if client is not None else _build_client()
        self.requests = 0

    def __call__(self, history: Sequence[Message]) -> AsyncGenerator[str, None]:
        return chunk_into_sentences(self._tokens(history), max_chars=self.max_chunk_chars)

    async def _tokens(self, history: Sequence[Message]) -> AsyncGenerator[str, None]:
        """
        Raw token stream. Sentence chunking happens one layer up.

        The system prompt is a message here rather than a top-level argument -- the one
        shape difference from the Anthropic adapter that is not merely a renamed keyword.
        """
        turns: list[Message] = list(history) or [
            {"role": "user", "content": "[the candidate has joined and is ready to begin]"}
        ]
        self.requests += 1

        stream = await self._client.chat.completions.create(  # type: ignore[attr-defined]
            model=self.model,
            max_completion_tokens=self.max_completion_tokens,
            messages=[{"role": "system", "content": self.system}, *turns],
            stream=True,
        )
        # Two levels of close, for the same reason as the Anthropic adapter: `async for`
        # closes nothing, so each wrapper must propagate it or the one below is left
        # suspended with its HTTP response open.
        async with stream as live, aclosing(live.__aiter__()) as chunks:
            async for chunk in chunks:
                if not chunk.choices:
                    # Usage-only or filter-only chunks carry no choices at all. Indexing
                    # [0] unconditionally is the usual way this crashes in production.
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta


LOCAL_BASE_URL = "http://localhost:11434/v1"
"""
Ollama's OpenAI-compatible endpoint.

Worth stating plainly because it is the most useful property of this adapter: Ollama, LM
Studio, and vLLM all speak the OpenAI wire format, so running a local model needs **no
new adapter** -- only a base URL and a model name. That is the boundary paying for
itself, and it makes a zero-cost, zero-key, fully-offline interviewer a config change:

    AVATAR_LLM=openai \
    OPENAI_BASE_URL=http://localhost:11434/v1 \
    AVATAR_LLM_MODEL=llama3.2 \
    uvicorn avatar.server:app

A local endpoint needs no credential, so a placeholder key is supplied rather than
demanding one -- the SDK requires the field to be non-empty and never sends it anywhere
that checks.
"""


def _build_client() -> object:
    """Construct the async client, failing with a message that says where to put the key."""
    try:
        from openai import AsyncOpenAI
    except ModuleNotFoundError as exc:  # pragma: no cover - environment, not logic
        raise RuntimeError("the OpenAI LLM needs the SDK: pip install -e '.[llm]'") from exc

    base_url = os.environ.get("OPENAI_BASE_URL")
    key = os.environ.get("OPENAI_API_KEY")

    if base_url and not key:
        # A local server authenticates nothing. Requiring a key here would block the
        # cheapest way to run this whole system.
        return AsyncOpenAI(base_url=base_url, api_key="local-no-key-required")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Put it in .env (gitignored) and run with "
            "`set -a && . ./.env && set +a`, or export it.\n"
            f"For a free local model instead, run Ollama and set "
            f"OPENAI_BASE_URL={LOCAL_BASE_URL} with AVATAR_LLM_MODEL=<pulled model>.\n"
            "Or run with AVATAR_LLM=scripted to use the canned interviewer."
        )
    return AsyncOpenAI(base_url=base_url) if base_url else AsyncOpenAI()
