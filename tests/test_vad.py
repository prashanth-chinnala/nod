"""
The energy gate, and the two claims made about it.

`EnergyVad` is not a voice activity detector and the tests say so. What is worth pinning
is that it is monotonic in loudness, bounded to [0, 1], and produces probabilities the
turn policy's default thresholds actually cross — a gate whose output never reached 0.6
would make onset unreachable, and nothing else in the suite would notice.

`SileroVad` is not tested here. It has never been executed; it needs torch, which is
deliberately outside the CI dependency set. Its conformance to the Protocol is checked
structurally below without instantiating it.
"""

from __future__ import annotations

import math
import struct

import pytest

from avatar.audio.turn_detection import ONSET_PROBABILITY, TurnDetector
from avatar.audio.vad import (
    FRAME_BYTES,
    FRAME_MS,
    FRAME_SAMPLES,
    SAMPLE_RATE,
    EnergyVad,
    SileroVad,
    SpeechProbability,
    build_vad,
)

RENDERER_METHODS = ("__call__", "reset")


def frame(amplitude: float) -> bytes:
    """One VAD frame of a sine at the given peak amplitude."""
    peak = int(32767 * amplitude)
    step = 2 * math.pi * 220.0 / SAMPLE_RATE
    return struct.pack(
        f"<{FRAME_SAMPLES}h",
        *(int(peak * math.sin(step * n)) for n in range(FRAME_SAMPLES)),
    )


def silence() -> bytes:
    return b"\x00" * FRAME_BYTES


# -- frame geometry --------------------------------------------------------


def test_frame_size_matches_what_silero_requires() -> None:
    """
    512 samples at 16kHz is not a free choice.

    Silero is trained on a fixed window and silently scores a short frame badly rather
    than rejecting it, which presents as a VAD that misses quiet speech.
    """
    assert (SAMPLE_RATE, FRAME_SAMPLES) == (16_000, 512)
    assert FRAME_BYTES == 1024
    assert FRAME_MS == 32


# -- energy gate -----------------------------------------------------------


def test_silence_scores_zero() -> None:
    assert EnergyVad()(silence()) == 0.0


def test_loud_audio_scores_one() -> None:
    assert EnergyVad()(frame(0.5)) == 1.0


def test_output_is_monotonic_in_loudness() -> None:
    vad = EnergyVad()
    scores = [vad(frame(a)) for a in (0.004, 0.012, 0.04, 0.12, 0.4)]

    assert scores == sorted(scores)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_output_crosses_the_default_onset_threshold() -> None:
    """
    Otherwise onset is unreachable and nothing else in the suite would catch it.

    The turn policy's thresholds and this gate's dB range are tuned against each other;
    changing either in isolation silently breaks turn-taking.
    """
    vad = EnergyVad()

    assert vad(frame(0.02)) < ONSET_PROBABILITY, "conversational room noise stays under"
    assert vad(frame(0.2)) >= ONSET_PROBABILITY, "speech at a normal level gets over"


def test_a_short_frame_does_not_raise() -> None:
    """
    The gate is length-agnostic, unlike Silero.

    Worth knowing rather than relying on: the server buffers to a full frame either way,
    because the two implementations must be interchangeable.
    """
    assert 0.0 <= EnergyVad()(b"\x00\x10" * 10) <= 1.0


def test_reset_is_safe_and_stateless() -> None:
    vad = EnergyVad()
    before = vad(frame(0.1))
    vad.reset()

    assert vad(frame(0.1)) == before


def test_inverted_thresholds_are_rejected() -> None:
    with pytest.raises(ValueError, match="ceiling_dbfs"):
        EnergyVad(floor_dbfs=-20.0, ceiling_dbfs=-40.0)


# -- the gate driving the real policy --------------------------------------


def test_energy_gate_and_turn_policy_agree_on_a_synthetic_utterance() -> None:
    """
    The two halves wired together, which nothing else in the suite does.

    Loud frames for ~0.5s then silence should produce exactly one onset and one
    end-of-turn. If the dB range and the probability thresholds ever drift apart, this
    is the test that fails rather than the demo.
    """
    from avatar.audio.turn_detection import EventKind

    vad = EnergyVad()
    det = TurnDetector(frame_ms=FRAME_MS)
    events = []

    for _ in range(16):  # ~512ms of speech
        events.extend(det.push(vad(frame(0.25))))
    for _ in range(30):  # ~960ms of silence, past the 700ms default
        events.extend(det.push(vad(silence())))

    assert [e.kind for e in events] == [
        EventKind.SPEECH_START,
        EventKind.END_OF_TURN,
    ]
    assert events[1].speech_ms >= 400


# -- the registry and the unverified implementation ------------------------


def test_build_returns_the_energy_gate_by_default() -> None:
    assert isinstance(build_vad(), EnergyVad)
    assert isinstance(build_vad("energy"), EnergyVad)


def test_build_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown VAD"):
        build_vad("whisper-but-somehow-a-vad")


def test_energy_gate_satisfies_the_protocol() -> None:
    vad: SpeechProbability = EnergyVad()

    assert vad.sample_rate == SAMPLE_RATE
    assert vad.frame_samples == FRAME_SAMPLES


@pytest.mark.parametrize("method_name", RENDERER_METHODS)
def test_silero_declares_the_protocol_surface(method_name: str) -> None:
    """
    Structural check only. `SileroVad` is never instantiated here.

    It needs torch, which is outside the CI dependency set on purpose, and it has never
    been executed anywhere -- recorded as unverified in DEVLOG.md. What this asserts is
    that swapping the detector is a config change and not a rewrite.
    """
    assert callable(getattr(SileroVad, method_name, None))
    assert SileroVad.sample_rate == SAMPLE_RATE
    assert SileroVad.frame_samples == FRAME_SAMPLES
