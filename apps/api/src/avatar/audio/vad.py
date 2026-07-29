"""
Voice activity detection: PCM in, a speech probability out.

Everything interesting about turn-taking lives in `turn_detection`, not here. This
module's whole job is to answer "how likely is it that this 32ms of audio is speech?"
and the policy layer decides what that means. Keeping the split means thresholds can be
tested as a table of floats, and swapping the detector does not touch the policy.

Two implementations, the same relationship as `StubRenderer` and the real model:

`EnergyVad` is an RMS gate with no dependencies at all. It is **not** a voice activity
detector — it cannot tell speech from a slammed door, a fan, or music, because it only
knows how loud the frame is. It exists so the full path works on a clean clone and in
CI, and so the policy layer has something to consume without a 2GB download.

`SileroVad` is the real thing: a small trained model that distinguishes speech from
noise. It needs torch.

The honest state of the second one is in its docstring: it has never been executed.
"""

from __future__ import annotations

import math
from array import array
from typing import Protocol

SAMPLE_RATE = 16_000
"""
Both implementations assume 16kHz mono s16le, which is also what `ToneTTS` emits.

Silero is trained at 16kHz and 8kHz specifically; feeding it anything else silently
degrades it rather than failing, which is the worst kind of bug. Pinning the rate here
and asserting on it is cheaper than debugging that later.
"""

FRAME_SAMPLES = 512
"""
32ms at 16kHz.

Not a free choice: Silero expects exactly 512 samples per call at this rate. The energy
gate does not care, but matching it means the policy layer sees the same frame duration
either way and thresholds tuned against one carry over to the other.
"""

FRAME_MS = round(FRAME_SAMPLES / SAMPLE_RATE * 1000)
FRAME_BYTES = FRAME_SAMPLES * 2


class SpeechProbability(Protocol):
    """One frame of audio in, probability of speech out."""

    sample_rate: int
    frame_samples: int

    def __call__(self, pcm: bytes) -> float:
        """Probability in [0.0, 1.0]. Must accept exactly `frame_samples` samples."""
        ...

    def reset(self) -> None:
        """Clear any internal state between utterances. Must be safe to call anytime."""
        ...


def _rms_dbfs(pcm: bytes) -> float:
    """RMS of s16le mono samples, in dBFS. Returns -inf-ish for silence."""
    if len(pcm) < 2:
        return -120.0
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not samples:
        return -120.0
    total = 0
    for sample in samples:
        total += sample * sample
    rms = math.sqrt(total / len(samples)) / 32768.0
    return 20 * math.log10(rms) if rms > 1e-6 else -120.0


class EnergyVad:
    """
    An RMS gate, honestly labelled.

    Maps loudness to a pseudo-probability by interpolating between two dB thresholds.
    Works everywhere, needs nothing, and will happily report 1.0 for a passing lorry.

    Two consequences worth stating rather than discovering:

      - It cannot gate out the avatar's own voice. The client relies on the browser's
        echo cancellation for that, and if that fails the avatar interrupts itself in a
        loop. `SileroVad` does not fix this either — echo cancellation is the fix.
      - The floor is room-dependent. A quiet room and a noisy café need different
        values, and nothing here adapts. A real deployment estimates the noise floor
        over the first second and offsets from it.
    """

    sample_rate = SAMPLE_RATE
    frame_samples = FRAME_SAMPLES

    def __init__(self, *, floor_dbfs: float = -48.0, ceiling_dbfs: float = -26.0) -> None:
        if ceiling_dbfs <= floor_dbfs:
            raise ValueError("ceiling_dbfs must be above floor_dbfs")
        self.floor_dbfs = floor_dbfs
        self.ceiling_dbfs = ceiling_dbfs

    def __call__(self, pcm: bytes) -> float:
        level = _rms_dbfs(pcm)
        span = self.ceiling_dbfs - self.floor_dbfs
        return max(0.0, min(1.0, (level - self.floor_dbfs) / span))

    def reset(self) -> None:
        """Stateless, so nothing to clear. Present to satisfy the Protocol."""
        return None


class SileroVad:
    """
    Silero VAD via torch.hub.

    **This code has never been executed.** There is no torch in the development
    environment it was written in, so it is structurally complete and empirically
    unverified. It is excluded from CI, marked in the test suite, and recorded as
    unverified in `DEVLOG.md`. Treat the first run as part of the work, not a
    formality.

    Why it is written anyway: the boundary is the deliverable. The policy layer, the
    server wiring, the client, and the tests are all complete and verified against
    `EnergyVad`, and this class is the demonstration that swapping the detector is a
    one-line config change rather than a rewrite.

    Loading downloads ~2MB of weights and pulls in torch, which is why it lives behind
    an optional extra and a lazy import. Nothing imports this module's torch at package
    import time.
    """

    sample_rate = SAMPLE_RATE
    frame_samples = FRAME_SAMPLES

    def __init__(self, *, force_reload: bool = False) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - environment, not logic
            raise RuntimeError(
                "SileroVad needs torch: pip install -e '.[vad]'. "
                "The default EnergyVad needs nothing."
            ) from exc

        self._torch = torch
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=force_reload,
            trust_repo=True,
        )
        model.eval()
        self._model = model

    def __call__(self, pcm: bytes) -> float:
        if len(pcm) != FRAME_BYTES:
            # Silero is trained on a fixed window. A short frame would be silently
            # padded by the tensor conversion and score badly, which reads as a VAD
            # that misses quiet speech.
            raise ValueError(
                f"expected exactly {FRAME_BYTES} bytes ({FRAME_SAMPLES} samples), "
                f"got {len(pcm)}"
            )
        samples = array("h")
        samples.frombytes(pcm)
        tensor = self._torch.tensor([s / 32768.0 for s in samples], dtype=self._torch.float32)
        with self._torch.no_grad():
            return float(self._model(tensor, self.sample_rate).item())

    def reset(self) -> None:
        """
        Clear the model's recurrent state.

        Silero carries state across calls, so a new utterance must start clean or the
        tail of the previous one biases it.
        """
        reset = getattr(self._model, "reset_states", None)
        if reset is not None:
            reset()


def build_vad(name: str = "energy") -> SpeechProbability:
    """
    The one-line VAD swap, mirroring `renderers.build`.

    `AVATAR_VAD=silero uvicorn avatar.server:app` is the whole change. Lazy import so
    that choosing `energy` never touches torch.
    """
    key = name.lower()
    if key == "energy":
        return EnergyVad()
    if key == "silero":
        return SileroVad()
    raise ValueError(f"unknown VAD {name!r}; available: 'energy', 'silero'")
