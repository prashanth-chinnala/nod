#!/usr/bin/env python3
"""
The voice-cloning sidecar: one process, one model, one endpoint.

**Why this is a separate process rather than an import.** Chatterbox needs a `transformers`
recent enough to expose `LlamaModel`; MuseTalk pins `transformers==4.39.2`, which does not.
Installing both into one environment does not merely warn -- it downgraded numpy and torch, left
`libtorch_cuda.so: undefined symbol: ncclCommResume`, and took the renderer with it. There is no
version of `transformers` that satisfies both, so no amount of pinning fixes this.

Which turns out to be the better architecture anyway, for a reason that has nothing to do with
dependencies: the runtime's `SpeechStream` boundary already treats speech synthesis as something
that happens elsewhere. A hosted API and a local sidecar are the same shape to the caller. So
the server's environment stays free of a second ML stack, the two models can be upgraded
independently, and the GPU memory each holds is visible as a separate process rather than hidden
inside one.

**Run it in its own environment**, which is where Chatterbox is installed:

    .venv-voice/bin/python scripts/voice_service.py            # :8100 by default

The runtime finds it at `AVATAR_VOICE_SERVICE`, default `http://127.0.0.1:8100`. If it is not
running, `build_tts("clone")` fails with a message naming this file rather than a connection
error.

**It returns raw PCM, not WAV.** The caller chunks it into `AudioChunk`s and needs no header;
16-bit mono little-endian at 16kHz is what the whole pipeline speaks, so converting here keeps
that single assumption in one place.
"""

from __future__ import annotations

import io
import os
import time
import wave
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

SAMPLE_RATE = 16_000
"""What the pipeline speaks. The model produces 24kHz; resampled before returning."""

MAX_REFERENCE_SECONDS = 30.0
"""
How much of a reference recording is used for conditioning.

Uploads may be up to 120s so an operator need not trim a natural sample, but the speaker
embedding saturates long before that and every extra second is spent on the first
`prepare_conditionals` call.
"""

app = FastAPI(title="nod voice", docs_url=None, redoc_url=None)

_state: dict[str, Any] = {"model": None, "conditioned": ""}


class Ask(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    reference_path: str = Field(min_length=1)


def _model() -> Any:
    """Load once per process. ~33s, which is why it is not per request."""
    if _state["model"] is None:
        import torch
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            # Loud, because on CPU the real-time factor is above 1.0 -- which is not "slow", it
            # is
            # audio that falls further behind the conversation with every sentence.
            print(
                f"!! voice service on device={device!r}. Measured RTF is 0.74 on a GPU "
                "and well above 1.0 on CPU, where a turn never catches up.",
                flush=True,
            )
        started = time.perf_counter()
        _state["model"] = ChatterboxTurboTTS.from_pretrained(device=device)
        print(f"voice: model loaded in {time.perf_counter() - started:.1f}s", flush=True)
    return _state["model"]


def _trimmed(reference: str) -> str:
    """
    The reference cut to `MAX_REFERENCE_SECONDS`, cached beside the original.

    Written next to the source rather than in a temporary directory, so the trim happens once
    per
    voice for the life of the deployment rather than once per process.
    """
    import subprocess
    from pathlib import Path

    source = Path(reference)
    if not source.exists():
        raise HTTPException(status_code=404, detail=f"reference not found: {source}")
    target = source.with_name(f"{source.stem}-ref{int(MAX_REFERENCE_SECONDS)}s.wav")
    if not target.exists():
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-v", "error", "-y",
                "-i", str(source), "-t", str(MAX_REFERENCE_SECONDS),
                "-ac", "1", "-ar", "24000", str(target),
            ],
            check=True,
            timeout=120,
        )
    return str(target)


def _pcm16(wav: Any, model_rate: int) -> bytes:
    """
    Model output to 16kHz mono 16-bit PCM.

    Resampled with torchaudio rather than by dropping samples: 24k to 16k is 2:3 and does not
    land
    on integer samples, and doing it crudely is audible as a metallic edge on sibilants -- the
    kind
    of defect that gets blamed on the model.
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
    # Clamped before scaling: a model can overshoot [-1, 1], and int16 wraps rather than clips,
    # which is not a quiet distortion but a loud click.
    clamped = audio.squeeze(0).clamp(-1.0, 1.0)
    return bytes((clamped * 32767.0).to(torch.int16).numpy().tobytes())


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    """Whether the model is loaded, without loading it. Used by warm-up and by `/config`."""
    return {
        "ok": True,
        "loaded": _state["model"] is not None,
        "conditioned_on": _state["conditioned"],
        "sample_rate": SAMPLE_RATE,
    }


@app.post("/warm")
def warm() -> dict[str, Any]:
    """Load the model now, so the first interview does not pay 33 seconds for it."""
    started = time.perf_counter()
    _model()
    return {"loaded": True, "ms": round((time.perf_counter() - started) * 1000)}


@app.post("/synthesise")
def synthesise(ask: Ask) -> Response:
    """
    One sentence, in the voice of one reference. Returns raw PCM.

    Conditioning is cached by reference path: it is the expensive part of a first call and
    identical
    every time for a given voice. Switching voices mid-process re-conditions, which is correct
    and
    costs a second -- so a deployment serving two personas at once wants two of these, not one.
    """
    model = _model()
    reference = _trimmed(ask.reference_path)
    if _state["conditioned"] != reference:
        model.prepare_conditionals(reference, exaggeration=0.5)
        _state["conditioned"] = reference

    started = time.perf_counter()
    try:
        wav = model.generate(ask.text, audio_prompt_path=reference)
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"{type(exc).__name__}: {exc}"
        ) from exc
    pcm = _pcm16(wav, model.sr)
    seconds = len(pcm) / 2 / SAMPLE_RATE
    elapsed = time.perf_counter() - started
    # Logged per sentence because real-time factor is the number that decides whether this is
    # usable at all, and it moves with sentence length and GPU contention.
    print(
        f"voice: {elapsed * 1000:.0f} ms -> {seconds:.1f}s audio  "
        f"RTF {elapsed / max(seconds, 1e-6):.2f}",
        flush=True,
    )
    return Response(content=pcm, media_type="application/octet-stream")


@app.post("/audition")
def audition(ask: Ask) -> Response:
    """The same synthesis, wrapped in a WAV header so a browser can play it directly."""
    pcm = synthesise(ask).body
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm)
    return Response(content=buffer.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("AVATAR_VOICE_HOST", "127.0.0.1"),
        port=int(os.environ.get("AVATAR_VOICE_PORT", 8100)),
    )
