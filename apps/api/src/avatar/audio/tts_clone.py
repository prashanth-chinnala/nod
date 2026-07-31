"""
A `SpeechStream` that sounds like a specific person, cloned from a reference recording.

**Why this fits behind the existing boundary at all.** The orchestrator consumes a
`SpeechStream`: sentences in, `AudioChunk`s out, cancellable by closing the generator. Nothing
about that assumes a hosted API, so a local model is a new implementation rather than a change
to the interview loop. `with_pronunciation` still wraps it, barge-in still works by closing the
generator mid-sentence, and history truncation still counts acknowledged audio.

**The measurement that decided the model, and it was not latency.** Real-time factor is what
matters. Measured on a T4, cloning from a 60s reference:

    chatterbox base    2515-4137 ms/sentence   RTF 1.31-1.61
    chatterbox Turbo   1213-2051 ms/sentence   RTF 0.67-0.80

Above 1.0 the generator falls further behind the longer a turn runs, and no amount of buffering
recovers it -- so the base model is disqualified however good it sounds. Below 1.0 it keeps
ahead of playback, which means **only the first sentence of a turn exposes its latency**; every
later sentence is produced while the previous one is still audible. That is the property the
sentence-level design was built to exploit, and it turns "a cloned voice costs seconds per turn"
into "a cloned voice costs about 1.6s once per turn" -- 2.0s against Deepgram Aura's measured
0.38s.

**What it costs, stated rather than buried.** The first sentence of every turn is slower than
with a hosted voice. In exchange the audio never leaves the machine, which is the whole argument
for self-hosting, and the persona can sound like a real person.

**Why generation runs in a thread.** `generate()` is synchronous and takes over a second. On the
event loop it would stall every other session in the process and, worse, prevent the WebSocket
from delivering the audio it had already produced. `asyncio.to_thread` per sentence.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from typing import Any

from avatar.audio.tts import CHUNK_MS, SAMPLE_RATE
from avatar.contracts import AudioChunk

MODEL_SAMPLE_RATE = 24_000
"""
What Chatterbox produces. Resampled to `SAMPLE_RATE` before leaving this module.

The pipeline is 16kHz end to end -- the VAD expects it, the renderer's Whisper features expect
it, and the browser's AudioContext is created at it. Converting here rather than anywhere
downstream keeps that single assumption true, and downsampling 24k to 16k discards nothing a
voice needs.
"""

MAX_REFERENCE_SECONDS = 30.0
"""
How much of the reference recording is actually used for conditioning.

`avatar.media` accepts up to 120s so an operator can upload a natural sample without trimming
it, but the speaker embedding saturates long before that and every extra second is time spent on
the first `prepare_conditionals` call. Trimmed here rather than at upload, so the stored file
stays the thing the operator provided.
"""


def _to_16k_pcm(wav: Any, model_rate: int) -> bytes:
    """
    Model output to the mono 16-bit little-endian PCM the transport speaks.

    Resampling with torchaudio rather than by slicing samples: naive decimation from 24k to 16k
    is a
    2:3 ratio that does not land on integer samples, and doing it wrong is audible as a metallic
    edge on sibilants -- the kind of defect that gets blamed on the model.
    """
    import torch
    import torchaudio

    audio = wav.detach().to("cpu").float()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if model_rate != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, model_rate, SAMPLE_RATE)
    # Clamped before scaling. A model can overshoot [-1, 1] by a little, and int16 wraps rather
    # than clipping -- which is not a quiet distortion, it is a loud click.
    clamped = audio.squeeze(0).clamp(-1.0, 1.0)
    return bytes((clamped * 32767.0).to(torch.int16).numpy().tobytes())


class ClonedTTS:
    """
    Chatterbox Turbo, conditioned on one reference recording.

    One model per process, shared across sessions: it is ~350M parameters and 32.9s to load, so
    a
    copy per session would be both slow and a way to exhaust a GPU that is also holding the
    renderer. The conditioning is per reference and cached, because it is the expensive part of
    the
    first call and identical every time for a given voice.
    """

    _model: Any = None
    """Class-level. See the docstring: one model per process, not per instance."""

    _conditioned: str = ""
    """Which reference the shared model is currently conditioned on."""

    def __init__(self, *, reference_path: str = "", chunk_ms: int = CHUNK_MS) -> None:
        self.reference_path = reference_path or os.environ.get("AVATAR_VOICE_REFERENCE", "")
        self.chunk_ms = chunk_ms
        self.sentences = 0
        self.generated_ms = 0

    # -- loading ------------------------------------------------------------

    def load(self) -> None:
        """
        Load the model. Called by warm-up, and lazily by the first sentence otherwise.

        Idempotent, and deliberately not in `__init__`: constructing this must stay cheap so
        `build_tts` can be called during configuration resolution without paying 33 seconds.
        """
        if ClonedTTS._model is not None:
            return
        import torch
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            # Loud rather than quietly unusable. RTF on CPU is far above 1, which means a turn
            # falls behind and never recovers -- a broken interview rather than a slow one.
            print(
                f"!! voice cloning on device={device!r}. Measured RTF on a GPU is 0.74; "
                "on CPU it is well above 1.0, which does not merely sound slow -- the "
                "audio falls further behind the conversation with every sentence.",
                flush=True,
            )
        ClonedTTS._model = ChatterboxTurboTTS.from_pretrained(device=device)

    def _prepare(self) -> Any:
        """Return the shared model, conditioned on this instance's reference."""
        self.load()
        model = ClonedTTS._model
        if ClonedTTS._conditioned != self.reference_path:
            # Trimmed to the useful length. `prepare_conditionals` walks the whole file
            # otherwise.
            model.prepare_conditionals(self._trimmed(), exaggeration=0.5)
            ClonedTTS._conditioned = self.reference_path
        return model

    def _trimmed(self) -> str:
        """
        The reference, cut to `MAX_REFERENCE_SECONDS`, cached beside the original.

        Written next to the source rather than into a temporary directory so the trim happens
        once
        per voice for the life of the deployment, not once per process.
        """
        import subprocess
        from pathlib import Path

        source = Path(self.reference_path)
        if not source.exists():
            raise RuntimeError(f"voice reference not found: {source}")
        target = source.with_name(f"{source.stem}-ref{int(MAX_REFERENCE_SECONDS)}s.wav")
        if target.exists():
            return str(target)
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-v", "error", "-y",
                "-i", str(source),
                "-t", str(MAX_REFERENCE_SECONDS),
                "-ac", "1", "-ar", str(MODEL_SAMPLE_RATE),
                str(target),
            ],
            check=True,
            timeout=120,
        )
        return str(target)

    # -- contract -----------------------------------------------------------

    def __call__(self, text: str, epoch: int) -> AsyncGenerator[AudioChunk, None]:
        return self._generate(text, epoch)

    async def _generate(self, text: str, epoch: int) -> AsyncGenerator[AudioChunk, None]:
        """
        One sentence in, a run of `AudioChunk`s out.

        The whole sentence is synthesised before the first chunk is yielded, which is a real
        difference from a streaming API and the reason first-sentence latency is what it is. It
        is
        also why the chunking below matters: the epoch is checked between chunks, so barge-in
        still
        interrupts *within* a sentence even though generation was atomic.
        """
        if not text.strip():
            return

        model = await asyncio.to_thread(self._prepare)
        wav = await asyncio.to_thread(model.generate, text, audio_prompt_path=self._trimmed())
        pcm = await asyncio.to_thread(_to_16k_pcm, wav, model.sr)

        self.sentences += 1
        self.generated_ms += len(pcm) // 2 * 1000 // SAMPLE_RATE

        step = self.chunk_ms * SAMPLE_RATE // 1000 * 2
        for offset in range(0, len(pcm), step):
            piece = pcm[offset : offset + step]
            if not piece:
                break
            # Duration from the byte count, not from `chunk_ms`: the final chunk of a sentence
            # is
            # short, and reporting a full chunk for it would inflate what the orchestrator
            # believes
            # was heard -- which is what history truncation trusts.
            yield AudioChunk(
                pcm=piece,
                epoch=epoch,
                duration_ms=len(piece) // 2 * 1000 // SAMPLE_RATE,
            )
            # Yield control between chunks. Without it a long sentence hands the loop a burst of
            # chunks with no chance to notice the socket closing or the turn being cancelled.
            await asyncio.sleep(0)

    async def aclose(self) -> None:
        """
        Release nothing. The model is process-wide and shared.

        Present because the protocol has it and a caller should not have to know which
        implementations own resources. Unloading here would make the next session pay 33s.
        """
        return None
