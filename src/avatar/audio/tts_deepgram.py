"""
Real speech: Deepgram Aura, streamed and chunked.

Implements `SpeechStream`, so nothing above it changes -- the orchestrator, the renderer,
the transport, and the client all keep working. `AVATAR_TTS=deepgram` is the switch.

Every number and format detail below was measured against the live API rather than read
off a docs page, because two of them would have produced broken audio if assumed.

**The RIFF header.** `/v1/speak` returns `audio/wav` by default -- a 44-byte
`RIFF....WAVEfmt` header before the samples. Feeding that straight into an `AudioChunk`
sends the header to the browser as PCM, which plays as a click at the start of every
sentence, and shifts the byte-to-duration arithmetic the renderer uses to drive the
mouth. `container=none` returns bare `audio/l16` instead, confirmed by inspecting the
first 16 bytes of both. That parameter is not optional.

**Measured latency**, `aura-2-thalia-en`, one sentence at a time, 16kHz mono:

| | |
|---|---|
| time-to-first-audio, warm | ~330-400ms (median 380ms over 9 runs across three lengths) |
| time-to-first-audio, cold | ~1020ms on the first request of a fresh connection |
| generation rate | ~0.6x realtime -- faster than playback, so the mixer's queue stays bounded |

Two consequences worth naming. The warm figure is above the 100-300ms band §1.5 budgets
for this stage, which is a real finding and not a rounding error. And the cold figure is
nearly 3x the warm one, which is why the HTTP client is constructed once and reused
rather than per request -- a fresh TLS handshake per sentence would put every turn on the
cold path.

**Sentence-at-a-time matters here.** TTFB was essentially flat across 48, 101, and 132
character inputs, so the win from sentence chunking is not a shorter first synthesis --
it is that synthesis of sentence two overlaps playback of sentence one. That is the
pipelining argument in §1.4, and it is why this adapter is fed one sentence at a time
rather than a whole response.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import aclosing

from avatar.audio.tts import CHUNK_MS, SAMPLE_RATE
from avatar.contracts import AudioChunk

SPEAK_URL = "https://api.deepgram.com/v1/speak"

DEFAULT_VOICE = "aura-2-thalia-en"
"""
An interviewer voice. Swap with `AVATAR_TTS_VOICE`.

Deliberately not a decision the code makes permanently: voice choice is a product call,
and the whole catalogue is one parameter away.
"""

BYTES_PER_SAMPLE = 2

REQUEST_TIMEOUT_S = 20.0
"""
Generous. A timeout that fires mid-sentence produces a half-spoken turn, which is worse
than a slow one -- and the barge-in path already provides the fast way to abandon a turn.
"""


def _chunk_bytes(chunk_ms: int) -> int:
    """Bytes of 16-bit mono PCM in `chunk_ms` of audio, at the module's sample rate."""
    return int(SAMPLE_RATE * chunk_ms / 1000) * BYTES_PER_SAMPLE


def duration_ms(pcm: bytes) -> int:
    """Playback duration of a 16-bit mono PCM buffer. The transport schedules on this."""
    return round(len(pcm) / BYTES_PER_SAMPLE / SAMPLE_RATE * 1000)


class DeepgramTTS:
    """
    A `SpeechStream` backed by Aura.

    Holds one `httpx.AsyncClient` for the process lifetime. That is not premature
    optimisation -- measured cold TTFB is ~1020ms against ~380ms warm, so a client per
    request would put every single turn on the slow path.

    Cancellation is the same contract as the LLM adapters: closing the generator unwinds
    the streaming response and aborts the request, so a barge-in stops paying for audio
    nobody will hear.
    """

    def __init__(
        self,
        *,
        voice: str | None = None,
        chunk_ms: int = CHUNK_MS,
        api_key: str | None = None,
        client: object | None = None,
    ) -> None:
        self.voice = voice or os.environ.get("AVATAR_TTS_VOICE", DEFAULT_VOICE)
        self.chunk_ms = chunk_ms
        self.chunk_bytes = _chunk_bytes(chunk_ms)
        self._api_key = api_key or os.environ.get("DEEPGRAM_API_KEY", "")
        self.requests = 0
        self.bytes_received = 0

        if client is not None:
            self._client = client
        else:
            if not self._api_key:
                raise RuntimeError(
                    "DEEPGRAM_API_KEY is not set. Put it in .env (gitignored) and run "
                    "with `set -a && . ./.env && set +a`, or export it. "
                    "Run with AVATAR_TTS=tone to use the placeholder synthesiser."
                )
            self._client = _build_client()

    @property
    def params(self) -> dict[str, str]:
        """
        Query parameters. `container=none` is load-bearing.

        Without it the response is `audio/wav` and the first 44 bytes are a RIFF header
        that would be played as audio and would corrupt the byte-to-duration arithmetic.
        """
        return {
            "model": self.voice,
            "encoding": "linear16",
            "sample_rate": str(SAMPLE_RATE),
            "container": "none",
        }

    def __call__(self, text: str, epoch: int) -> AsyncGenerator[AudioChunk, None]:
        return self._generate(text, epoch)

    async def _generate(self, text: str, epoch: int) -> AsyncGenerator[AudioChunk, None]:
        if not text.strip():
            # The API rejects empty input, and an empty sentence is a chunker artefact
            # rather than something worth a round trip.
            return

        self.requests += 1
        request = self._client.stream(  # type: ignore[attr-defined]
            "POST",
            SPEAK_URL,
            params=self.params,
            headers={
                "Authorization": f"Token {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"text": text},
        )

        buffer = bytearray()
        async with request as response:
            if response.status_code != 200:
                body = await response.aread()
                raise RuntimeError(
                    f"Deepgram TTS returned {response.status_code}: "
                    f"{body[:200].decode('utf-8', 'replace')}"
                )
            # `async for` closes nothing, so the byte iterator is closed explicitly --
            # the same three-level close chain the LLM adapters needed.
            async with aclosing(response.aiter_bytes()) as parts:
                async for part in parts:
                    self.bytes_received += len(part)
                    buffer.extend(part)
                    # Emit fixed-size chunks as they complete rather than waiting for the
                    # response to finish. Waiting would make first-audio equal to full
                    # synthesis time and defeat the point of streaming.
                    while len(buffer) >= self.chunk_bytes:
                        pcm = bytes(buffer[: self.chunk_bytes])
                        del buffer[: self.chunk_bytes]
                        yield AudioChunk(pcm=pcm, epoch=epoch, duration_ms=duration_ms(pcm))

        if buffer:
            # The tail is shorter than a full chunk. Emitting it matters: dropping it
            # would clip the end of every sentence, and `duration_ms` keeps the
            # transport's timeline honest about the short chunk.
            pcm = bytes(buffer)
            yield AudioChunk(pcm=pcm, epoch=epoch, duration_ms=duration_ms(pcm))

    async def aclose(self) -> None:
        """Release the shared connection pool. Called on server shutdown."""
        closer = getattr(self._client, "aclose", None)
        if closer is not None:
            await closer()


def _build_client() -> object:
    try:
        import httpx
    except ModuleNotFoundError as exc:  # pragma: no cover - environment, not logic
        raise RuntimeError("Deepgram TTS needs httpx: pip install -e '.[tts]'") from exc
    # One client for the process. See the class docstring: cold TTFB is ~3x warm.
    return httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S)


def build_tts(name: str = "tone") -> object:
    """
    The one-line TTS swap, mirroring the renderer, VAD, and LLM registries.

    Defaults to `tone` so a clean clone runs with no credentials and no network.
    """
    key = name.lower()
    if key == "tone":
        from avatar.audio.tts import ToneTTS

        return ToneTTS()
    if key == "deepgram":
        return DeepgramTTS()
    raise ValueError(f"unknown TTS {name!r}; available: 'tone', 'deepgram'")
