"""
The Claude adapter, and the cancellation guarantee it depends on.

No network. A fake client stands in for the SDK, which lets the request shape and the
abort-on-cancel behaviour be asserted exactly — and keeps the suite runnable on a clean
clone with no key.

The abort test is the one that matters. Returning out of an `async for` does not close
the generator; Python defers that to the garbage collector. For an HTTP-backed LLM that
means a barge-in stops *reading* the response while the provider keeps generating and
billing it. `SessionOrchestrator._run_turn` closes deterministically with `aclosing`, and
this file proves the adapter's `finally` actually runs when it does.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

import pytest

from avatar.contracts import Message
from avatar.llm import ScriptedInterviewer
from avatar.llm_anthropic import (
    INTERVIEWER_SYSTEM,
    AnthropicInterviewer,
    build_llm,
)
from avatar.orchestrator import SessionOrchestrator
from avatar.state import State
from tests.conftest import ScriptedTTS, run_until, settle


class FakeUsage:
    input_tokens = 120
    output_tokens = 34


class FakeFinal:
    stop_reason = "end_turn"
    usage = FakeUsage()


class FakeStream:
    """Stands in for the SDK's streaming context manager."""

    def __init__(self, owner: FakeMessages, tokens: list[str], gate: asyncio.Event | None):
        self._owner = owner
        self._tokens = tokens
        self._gate = gate

    async def __aenter__(self) -> FakeStream:
        self._owner.entered += 1
        return self

    async def __aexit__(self, *exc: object) -> bool:
        self._owner.exited += 1
        return False

    @property
    def text_stream(self) -> AsyncIterator[str]:
        return self._emit()

    async def _emit(self) -> AsyncIterator[str]:
        try:
            for i, token in enumerate(self._tokens):
                if self._gate is not None and i == 1:
                    # Park mid-response so a test can abandon the turn here.
                    await self._gate.wait()
                yield token
                await asyncio.sleep(0)
        finally:
            self._owner.generator_closed = True

    async def get_final_message(self) -> FakeFinal:
        return FakeFinal()


class FakeMessages:
    def __init__(self, tokens: list[str], gate: asyncio.Event | None = None) -> None:
        self._tokens = tokens
        self._gate = gate
        self.calls: list[dict[str, object]] = []
        self.entered = 0
        self.exited = 0
        self.generator_closed = False

    def stream(self, **kwargs: object) -> FakeStream:
        self.calls.append(kwargs)
        return FakeStream(self, self._tokens, self._gate)


class FakeClient:
    def __init__(self, tokens: list[str], gate: asyncio.Event | None = None) -> None:
        self.messages = FakeMessages(tokens, gate)


async def collect(source: AsyncIterator[str]) -> list[str]:
    return [item async for item in source]


# -- request shape ---------------------------------------------------------


async def test_streams_and_chunks_into_sentences() -> None:
    client = FakeClient(["Tell me", " about a failure.", " What broke", "?"])
    llm = AnthropicInterviewer(client=client)

    assert await collect(llm([])) == ["Tell me about a failure.", " What broke?"]


async def test_the_request_omits_sampling_parameters() -> None:
    """`temperature`, `top_p`, and `top_k` are rejected outright on this model."""
    client = FakeClient(["Hello."])
    await collect(AnthropicInterviewer(client=client)([]))

    sent = client.messages.calls[0]
    assert not {"temperature", "top_p", "top_k"} & sent.keys()


async def test_thinking_is_disabled_at_an_effort_that_permits_it() -> None:
    """
    Thinking tokens are generated before any text, so they land on TTFT directly.

    Disabling is only accepted at effort `high` or below on this model — pairing it with
    `xhigh` or `max` is a 400, so the two settings are checked together.
    """
    client = FakeClient(["Hello."])
    await collect(AnthropicInterviewer(client=client)([]))

    sent = client.messages.calls[0]
    assert sent["thinking"] == {"type": "disabled"}
    assert sent["output_config"] == {"effort": "low"}


def test_the_system_prompt_carries_the_tag_leakage_mitigation() -> None:
    """
    Disabling thinking can leak internal tags into the visible response.

    The mitigation must be phrased generically rather than naming thinking tags — the
    named form is measurably less effective — and must not contain a "do not think"
    rule, which makes leakage worse.
    """
    lowered = INTERVIEWER_SYSTEM.lower()

    assert "internal or system xml tags" in lowered
    assert "<thinking>" not in lowered
    assert "do not think" not in lowered


async def test_an_empty_history_still_sends_a_user_turn() -> None:
    """The API requires the first message to be from the user."""
    client = FakeClient(["Hello."])
    await collect(AnthropicInterviewer(client=client)([]))

    messages = client.messages.calls[0]["messages"]
    assert isinstance(messages, list) and messages[0]["role"] == "user"


async def test_history_is_forwarded_unchanged() -> None:
    history: list[Message] = [
        {"role": "user", "content": "the billing migration"},
        {"role": "assistant", "content": "What broke? [interrupted]"},
    ]
    client = FakeClient(["Go on."])

    await collect(AnthropicInterviewer(client=client)(history))

    assert client.messages.calls[0]["messages"] == history


async def test_usage_is_recorded_for_the_cost_model() -> None:
    client = FakeClient(["Hello."])
    llm = AnthropicInterviewer(client=client)

    await collect(llm([]))

    assert llm.last_usage == {"input_tokens": 120, "output_tokens": 34}
    assert llm.requests == 1


# -- cancellation aborts the upstream request ------------------------------


async def test_closing_the_generator_exits_the_streaming_context() -> None:
    """
    The abort guarantee, at the adapter level.

    Exiting the context manager is what aborts the HTTP request. If this stops holding,
    a barge-in leaves the provider generating a response nobody will hear.
    """
    gate = asyncio.Event()
    # Token 0 must close a sentence. The chunker only yields on a terminator, so a
    # token that does not end one would leave `anext` blocked on the gate forever.
    client = FakeClient(["Tell me about a failure.", " What broke?"], gate=gate)
    llm = AnthropicInterviewer(client=client)

    stream = llm([])
    assert await anext(stream) == "Tell me about a failure."
    assert client.messages.entered == 1
    assert client.messages.exited == 0

    await stream.aclose()

    assert client.messages.generator_closed is True
    assert client.messages.exited == 1, "the streaming context must be exited on close"


async def test_a_barge_in_aborts_the_llm_request(
    build_session: Callable[..., SessionOrchestrator],
) -> None:
    """
    The same guarantee, end to end through the orchestrator.

    This is what `aclosing` in `_run_turn` buys. Without it the generator is left to the
    garbage collector, and this assertion fails intermittently — the worst possible
    shape for a billing bug.
    """
    gate = asyncio.Event()
    # Token 0 must close a sentence, or nothing reaches the TTS and the session
    # never reaches SPEAKING.
    client = FakeClient(["A long answer that gets cut off.", " And more."], gate=gate)
    llm = AnthropicInterviewer(client=client)
    orch = build_session(llm=llm, tts=ScriptedTTS(chunks_per_sentence=12))

    await orch.start("reference.mp4")
    await orch.on_speech_start()
    await orch.on_end_of_turn("tell me about a failure")
    await run_until(lambda: orch.state is State.SPEAKING, what="SPEAKING")

    await orch.on_speech_start()  # barge-in
    gate.set()
    await settle(orch)

    assert orch.state is State.LISTENING
    assert client.messages.exited == 1, "the abandoned request was never aborted"


# -- the registry ----------------------------------------------------------


def test_build_defaults_to_the_scripted_interviewer() -> None:
    """A clean clone must run with no key and no network."""
    assert isinstance(build_llm(), ScriptedInterviewer)
    assert isinstance(build_llm("scripted"), ScriptedInterviewer)


def test_build_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown LLM"):
        build_llm("some-model-that-does-not-exist")


def test_missing_key_fails_with_an_actionable_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The most common misconfiguration, and the SDK's own error does not mention where
    this project expects the key to live.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        build_llm("anthropic")
