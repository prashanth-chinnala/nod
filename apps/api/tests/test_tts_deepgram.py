"""
The Aura adapter. No network — a fake httpx stands in for the wire.

Two of these tests exist because the live API surprised me, and an assumption in either
place would have shipped broken audio:

  - `/v1/speak` returns `audio/wav` by default. The 44-byte RIFF header would be played
    to the candidate as PCM (a click at the start of every sentence) and would skew the
    byte-to-duration arithmetic the renderer uses to drive the mouth. `container=none` is
    the fix and is asserted here so it cannot be dropped.
  - Chunk durations must come from byte length, not from a constant. Sentence tails are
    shorter than a full chunk, and a fixed duration would drift the transport's timeline
    a little on every sentence.
"""

from __future__ import annotations

import pytest

from avatar.audio.tts import CHUNK_MS, SAMPLE_RATE, ToneTTS
from avatar.audio.tts_deepgram import (
    DeepgramTTS,
    build_tts,
    duration_ms,
)

CHUNK_BYTES = int(SAMPLE_RATE * CHUNK_MS / 1000) * 2  # 80ms of 16kHz mono s16le = 2560


class FakeResponse:
    def __init__(self, owner: FakeClient, parts: list[bytes], status_code: int) -> None:
        self._owner = owner
        self._parts = parts
        self.status_code = status_code

    async def __aenter__(self) -> FakeResponse:
        self._owner.entered += 1
        return self

    async def __aexit__(self, *exc: object) -> bool:
        self._owner.exited += 1
        return False

    async def aread(self) -> bytes:
        return b'{"err_msg":"nope"}'

    async def aiter_bytes(self):
        try:
            for part in self._parts:
                yield part
        finally:
            self._owner.iterator_closed = True


class FakeClient:
    def __init__(self, parts: list[bytes], status_code: int = 200) -> None:
        self._parts = parts
        self._status = status_code
        self.calls: list[dict[str, object]] = []
        self.entered = 0
        self.exited = 0
        self.iterator_closed = False
        self.aclosed = False

    def stream(self, method: str, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        return FakeResponse(self, self._parts, self._status)

    async def aclose(self) -> None:
        self.aclosed = True


def tts(parts: list[bytes], status_code: int = 200) -> tuple[DeepgramTTS, FakeClient]:
    client = FakeClient(parts, status_code)
    return DeepgramTTS(api_key="test-key", client=client), client


async def collect(source):
    return [item async for item in source]


# -- the two format facts that came from measuring, not reading ------------


def test_container_none_is_requested() -> None:
    """
    Without it the response is `audio/wav` and the first 44 bytes are a RIFF header.

    Those bytes would reach the browser as PCM — an audible click on every sentence —
    and would corrupt the byte-to-duration arithmetic driving the mouth.
    """
    engine, _ = tts([b"\x00" * CHUNK_BYTES])

    assert engine.params["container"] == "none"


def test_the_requested_format_matches_what_the_pipeline_expects() -> None:
    engine, _ = tts([b"\x00" * CHUNK_BYTES])

    assert engine.params["encoding"] == "linear16"
    assert engine.params["sample_rate"] == str(SAMPLE_RATE)


def test_duration_comes_from_byte_length_not_a_constant() -> None:
    """A sentence tail is shorter than a chunk; a fixed duration would drift the timeline."""
    assert duration_ms(b"\x00" * CHUNK_BYTES) == CHUNK_MS
    assert duration_ms(b"\x00" * (CHUNK_BYTES // 2)) == CHUNK_MS // 2
    assert duration_ms(b"") == 0


# -- chunking --------------------------------------------------------------


async def test_bytes_are_emitted_as_fixed_size_chunks_while_streaming() -> None:
    """
    Waiting for the response to finish would make first-audio equal full synthesis time.

    Measured cold synthesis of one sentence was ~2s against ~440ms to first chunk, so
    buffering would cost roughly 1.5s of the latency budget per turn.
    """
    engine, _ = tts([b"\x01" * (CHUNK_BYTES * 2)])

    chunks = await collect(engine("Two whole chunks.", epoch=3))

    assert [len(c.pcm) for c in chunks] == [CHUNK_BYTES, CHUNK_BYTES]
    assert [c.duration_ms for c in chunks] == [CHUNK_MS, CHUNK_MS]


async def test_a_chunk_is_assembled_across_network_parts() -> None:
    """Network framing has nothing to do with audio framing."""
    engine, _ = tts([b"\x01" * 500, b"\x02" * 500, b"\x03" * (CHUNK_BYTES - 1000)])

    chunks = await collect(engine("One chunk in three packets.", epoch=1))

    assert len(chunks) == 1
    assert len(chunks[0].pcm) == CHUNK_BYTES


async def test_the_short_tail_is_emitted_not_dropped() -> None:
    """Dropping it would clip the end of every single sentence."""
    engine, _ = tts([b"\x01" * (CHUNK_BYTES + 800)])

    chunks = await collect(engine("A chunk and a bit.", epoch=1))

    assert [len(c.pcm) for c in chunks] == [CHUNK_BYTES, 800]
    assert chunks[-1].duration_ms == duration_ms(b"\x00" * 800)


async def test_the_epoch_is_propagated_to_every_chunk() -> None:
    """This is what lets a barge-in invalidate audio it has already handed downstream."""
    engine, _ = tts([b"\x01" * (CHUNK_BYTES * 3)])

    chunks = await collect(engine("Three chunks.", epoch=42))

    assert {c.epoch for c in chunks} == {42}


async def test_empty_text_makes_no_request() -> None:
    """An empty sentence is a chunker artefact, and the API rejects it anyway."""
    engine, client = tts([b"\x00" * CHUNK_BYTES])

    assert await collect(engine("   ", epoch=1)) == []
    assert client.calls == []


# -- failure and cancellation ---------------------------------------------


async def test_a_non_200_raises_with_the_body_included() -> None:
    """
    The orchestrator turns this into a session_failure and returns to IDLE.

    Including the body matters: Deepgram's 4xx messages name the actual problem, and
    swallowing them leaves only a status code to debug from.
    """
    engine, _ = tts([], status_code=402)

    with pytest.raises(RuntimeError, match="402"):
        await collect(engine("Anything.", epoch=1))


async def test_closing_mid_stream_exits_the_response_context() -> None:
    """
    Exiting the context is what aborts the HTTP request.

    Same guarantee as the LLM adapters: a barge-in must stop paying for audio nobody
    will hear, not merely stop reading it.
    """
    engine, client = tts([b"\x01" * CHUNK_BYTES] * 5)

    stream = engine("A long sentence we abandon.", epoch=1)
    first = await anext(stream)
    assert len(first.pcm) == CHUNK_BYTES
    assert client.exited == 0

    await stream.aclose()

    assert client.iterator_closed is True
    assert client.exited == 1, "the response context must be exited on close"


async def test_aclose_releases_the_shared_connection_pool() -> None:
    engine, client = tts([b"\x00" * CHUNK_BYTES])

    await engine.aclose()

    assert client.aclosed is True


# -- config and registry ---------------------------------------------------


def test_a_missing_key_fails_with_an_actionable_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY"):
        DeepgramTTS()


def test_the_voice_is_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AVATAR_TTS_VOICE", "aura-2-orion-en")

    engine = DeepgramTTS(api_key="k", client=FakeClient([]))

    assert engine.params["model"] == "aura-2-orion-en"


def test_build_defaults_to_the_placeholder_synthesiser() -> None:
    """A clean clone must run with no key and no network."""
    assert isinstance(build_tts(), ToneTTS)
    assert isinstance(build_tts("tone"), ToneTTS)


def test_build_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown TTS"):
        build_tts("some-engine-that-does-not-exist")
