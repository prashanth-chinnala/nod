"""
A `SpeechStream` that sounds like a specific person, served by the voice sidecar.

**Why this is an HTTP client and not an import.** The obvious version of this file loaded
Chatterbox in-process. It cannot: Chatterbox needs a `transformers` recent enough to expose
`LlamaModel`, and MuseTalk pins `transformers==4.39.2`, which does not. Installing both into one
environment downgraded numpy and torch and left `libtorch_cuda.so: undefined symbol:
ncclCommResume` -- it took the renderer down, not just the voice. No pin satisfies both, so this
is a process boundary rather than a dependency problem to solve.

Which is the better shape regardless of that collision, because the `SpeechStream` contract was
already written as though synthesis happened elsewhere -- a hosted API and a local sidecar look
identical to the orchestrator. The consequences all point the same way: the runtime's
environment stays free of a second ML stack, the two models upgrade independently, and each
one's GPU memory is a separate process an operator can see rather than a number hidden inside
one.

**The measurement that chose the model, and it is not latency.** Real-time factor. Measured on a
T4, cloning from a 60s reference:

    chatterbox base    2515-4137 ms/sentence   RTF 1.31-1.61
    chatterbox Turbo   1213-2051 ms/sentence   RTF 0.67-0.80

Above 1.0 the generator falls further behind the longer a turn runs and no buffering recovers
it, so the base model is disqualified however good it sounds. Below 1.0 it keeps ahead of
playback, so **only the first sentence of a turn exposes its latency** -- every later sentence
is produced while the previous one is still audible. That is the property the sentence-level
design exists to exploit, and it makes the honest cost "about 1.6s once per turn" rather than a
multiplier on the whole turn: 2.0s against Deepgram Aura's measured 0.38s.

Run the sidecar from its own environment, where Chatterbox lives:

    .venv-voice/bin/python scripts/voice_service.py
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from typing import Any

from avatar.audio.tts import CHUNK_MS, SAMPLE_RATE
from avatar.contracts import AudioChunk

SERVICE_ENV = "AVATAR_VOICE_SERVICE"
DEFAULT_SERVICE = "http://127.0.0.1:8100"

REQUEST_TIMEOUT_S = 60.0
"""
Generous, because a cold sidecar loads a 350M model on the first request -- 33s measured.

Not unbounded, though: a hung request would stall the turn indefinitely, where one that fails
loudly can at least be reported. The orchestrator treats an exception here as it treats any
other TTS failure.
"""


class ClonedTTS:
    """
    Speech in a cloned voice, one sentence per request to the sidecar.

    Holds one `httpx.AsyncClient` for its lifetime, for the same reason `DeepgramTTS` does: a
    client
    per request pays connection setup on every sentence of every turn.
    """

    def __init__(
        self,
        *,
        reference_path: str = "",
        service: str = "",
        chunk_ms: int = CHUNK_MS,
    ) -> None:
        self.reference_path = reference_path or os.environ.get("AVATAR_VOICE_REFERENCE", "")
        self.service = (service or os.environ.get(SERVICE_ENV, DEFAULT_SERVICE)).rstrip("/")
        self.chunk_ms = chunk_ms
        self.sentences = 0
        self.generated_ms = 0
        self._client: Any = None

    def _http(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S)
        return self._client

    # -- contract -----------------------------------------------------------

    def __call__(self, text: str, epoch: int) -> AsyncGenerator[AudioChunk, None]:
        return self._generate(text, epoch)

    async def _generate(self, text: str, epoch: int) -> AsyncGenerator[AudioChunk, None]:
        """
        One sentence in, a run of `AudioChunk`s out.

        The whole sentence is synthesised before the first chunk is yielded -- a real difference
        from a streaming API, and the reason first-sentence latency is what it is. Which is why
        the
        chunking below matters: the epoch travels on every chunk, so barge-in still discards
        audio
        from an abandoned turn *within* a sentence even though generation was atomic.
        """
        if not text.strip():
            return
        if not self.reference_path:
            raise RuntimeError(
                "the cloned voice has no reference recording. Attach a voice to the agent."
            )

        import httpx

        try:
            response = await self._http().post(
                f"{self.service}/synthesise",
                json={"text": text, "reference_path": self.reference_path},
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"the voice service at {self.service} is unreachable: {exc}. Start it with "
                "`.venv-voice/bin/python scripts/voice_service.py` -- it runs in its own "
                "environment because Chatterbox and MuseTalk cannot share one."
            ) from exc
        if response.status_code != 200:
            raise RuntimeError(
                f"the voice service refused this sentence ({response.status_code}): "
                f"{response.text[:200]}"
            )

        pcm = response.content
        self.sentences += 1
        self.generated_ms += len(pcm) // 2 * 1000 // SAMPLE_RATE

        step = self.chunk_ms * SAMPLE_RATE // 1000 * 2
        for offset in range(0, len(pcm), step):
            piece = pcm[offset : offset + step]
            if not piece:
                break
            # Duration from the byte count rather than from `chunk_ms`: the last chunk of a
            # sentence is short, and reporting a full one would inflate what the orchestrator
            # believes was heard -- which is exactly what history truncation trusts.
            yield AudioChunk(
                pcm=piece,
                epoch=epoch,
                duration_ms=len(piece) // 2 * 1000 // SAMPLE_RATE,
            )

    async def aclose(self) -> None:
        """Close the HTTP client. The model belongs to the sidecar and stays loaded."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
