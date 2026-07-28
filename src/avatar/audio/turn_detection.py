"""
Deciding when the candidate started talking, and when they stopped.

These are two different decisions with two different costs, and conflating them is what
makes an avatar either interrupt on a cough or talk over someone drawing breath. The
brief's own note on the latency budget calls end-of-turn detection "often the largest
and least-discussed term", so it gets its own module with its own tests.

**Speech onset** must be fast and must be certain, because acting on it interrupts the
avatar mid-sentence. A false positive is expensive and visible. So the bar is a high
probability sustained over several frames: ~100ms of delay bought in exchange for not
cutting the interviewer off every time someone's chair creaks.

**End of turn** must be patient. It fires when silence has persisted, and the threshold
is a *conversational* judgment, not a signal-processing one — people pause mid-sentence
to think, and an avatar that answers into that pause is worse than one that waits too
long. This term is pure configuration: whatever value is set here appears in the
latency budget as-is. It cannot be optimised away by faster hardware, which is exactly
why it deserves to be named rather than buried in a VAD's default.

**Retraction** exists because onset can be wrong. If speech was confirmed but the total
turned out to be shorter than a syllable, the right output is "never mind" rather than
an empty turn — which is why the orchestrator has an `on_speech_retract` at all.

Hysteresis between the two thresholds is the third piece: once speech is confirmed, it
takes a *lower* probability to stay in speech than it took to enter. Without it, the
dip in the middle of a word ends the turn.

This module consumes a stream of per-frame speech probabilities and knows nothing about
where they came from — no torch, no audio decoding, no model. That is what makes the
policy testable as a table of numbers, which is the only way to be confident about
thresholds without recording a corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

ONSET_PROBABILITY = 0.6
"""
How confident before we call it speech. Deliberately high.

Acting on this interrupts the avatar, so the cost of a false positive is a visible
stutter in the conversation. The cost of a false negative is ~40ms of extra delay.
"""

RELEASE_PROBABILITY = 0.35
"""
How confident to *stay* in speech. Lower than onset, on purpose.

The gap is hysteresis. Speech probability dips inside words -- plosives, the gap before
a stressed syllable -- and a single threshold turns each dip into an end-of-turn.
"""

ONSET_FRAMES = 3
"""
Consecutive frames above the onset bar before speech is declared.

At a 32ms frame this is ~96ms. It is the cheapest possible defence against a cough, a
door, or a keyboard: transient noise rarely sustains for three frames, and speech
always does.
"""

MIN_SPEECH_MS = 200
"""
Total speech below this is retracted rather than treated as a turn.

Shorter than a syllable is not an utterance, and handing it to the LLM produces a turn
built on nothing.
"""

END_OF_TURN_SILENCE_MS = 700
"""
Silence before the turn is considered finished.

The single largest term in the latency budget, and a conversational choice rather than
a technical one: too short and the avatar answers into a thinking pause, too long and
it feels sluggish. Whatever is set here shows up in the perceived turnaround unchanged
-- no amount of GPU makes it smaller.
"""


class EventKind(Enum):
    SPEECH_START = auto()
    SPEECH_RETRACT = auto()
    END_OF_TURN = auto()

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class TurnEvent:
    kind: EventKind
    at_ms: int
    """Detector time when the event fired, for correlating against telemetry."""
    speech_ms: int = 0
    """Speech accumulated in this turn. Zero for onset, meaningful at the end."""


class _Phase(Enum):
    QUIET = auto()
    CONFIRMING = auto()
    SPEAKING = auto()


class TurnDetector:
    """
    Turns per-frame speech probabilities into turn events.

    Deterministic and synchronous: `push` one frame's probability, get back the events
    that frame caused. No timers, no clock, no I/O — frame count *is* the clock, which
    means a test can express a scenario as a list of floats and get exact answers.
    """

    def __init__(
        self,
        *,
        frame_ms: int,
        onset_probability: float = ONSET_PROBABILITY,
        release_probability: float = RELEASE_PROBABILITY,
        onset_frames: int = ONSET_FRAMES,
        min_speech_ms: int = MIN_SPEECH_MS,
        end_of_turn_silence_ms: int = END_OF_TURN_SILENCE_MS,
    ) -> None:
        if frame_ms <= 0:
            raise ValueError("frame_ms must be positive")
        if onset_frames < 1:
            raise ValueError("onset_frames must be at least 1")
        if release_probability > onset_probability:
            # Not a style preference. A release bar above the onset bar inverts the
            # hysteresis and makes speech harder to sustain than to enter, which
            # produces an end-of-turn inside almost every word.
            raise ValueError(
                f"release_probability ({release_probability}) must not exceed "
                f"onset_probability ({onset_probability}); the gap is the hysteresis"
            )

        self.frame_ms = frame_ms
        self.onset_probability = onset_probability
        self.release_probability = release_probability
        self.onset_frames = onset_frames
        self.min_speech_ms = min_speech_ms
        self.end_of_turn_silence_ms = end_of_turn_silence_ms

        self._phase = _Phase.QUIET
        self._elapsed_ms = 0
        self._above_onset = 0
        self._speech_ms = 0
        self._silence_ms = 0

    # -- introspection, for the demo readout and for tests ------------------

    @property
    def in_speech(self) -> bool:
        return self._phase is _Phase.SPEAKING

    @property
    def speech_ms(self) -> int:
        return self._speech_ms

    @property
    def elapsed_ms(self) -> int:
        return self._elapsed_ms

    # -- the policy ---------------------------------------------------------

    def push(self, probability: float) -> list[TurnEvent]:
        """Feed one frame's speech probability. Returns any events it triggered."""
        self._elapsed_ms += self.frame_ms
        events: list[TurnEvent] = []

        if self._phase is _Phase.SPEAKING:
            # Hysteresis: staying in speech is easier than entering it.
            if probability >= self.release_probability:
                self._speech_ms += self.frame_ms
                self._silence_ms = 0
            else:
                self._silence_ms += self.frame_ms
                if self._silence_ms >= self.end_of_turn_silence_ms:
                    events.append(self._finish_turn())
            return events

        if probability >= self.onset_probability:
            self._above_onset += 1
            if self._above_onset >= self.onset_frames:
                self._phase = _Phase.SPEAKING
                # Credit the frames that established onset. They were speech; the
                # detector was only withholding judgment.
                self._speech_ms = self._above_onset * self.frame_ms
                self._silence_ms = 0
                self._above_onset = 0
                events.append(TurnEvent(EventKind.SPEECH_START, at_ms=self._elapsed_ms))
            else:
                self._phase = _Phase.CONFIRMING
        else:
            # A gap resets the run. Onset requires *consecutive* frames, or a noisy
            # room slowly accumulates its way past the threshold.
            self._above_onset = 0
            self._phase = _Phase.QUIET

        return events

    def _finish_turn(self) -> TurnEvent:
        spoken = self._speech_ms
        kind = (
            EventKind.END_OF_TURN if spoken >= self.min_speech_ms else EventKind.SPEECH_RETRACT
        )
        self._phase = _Phase.QUIET
        self._speech_ms = 0
        self._silence_ms = 0
        self._above_onset = 0
        return TurnEvent(kind, at_ms=self._elapsed_ms, speech_ms=spoken)

    def reset(self) -> None:
        """
        Forget the current turn without emitting anything.

        Called when the session state changes underneath the detector — after a
        barge-in, the turn it was tracking is no longer the turn in progress.
        """
        self._phase = _Phase.QUIET
        self._above_onset = 0
        self._speech_ms = 0
        self._silence_ms = 0
