"""
Turn-taking policy.

The detector consumes probabilities and nothing else, so a scenario is a list of floats
and every answer is exact. That is the reason for the split: thresholds are the kind of
thing normally tuned by ear against a recording, and here they can be tested.

Each test names the conversational failure it prevents, because the thresholds are only
defensible in terms of what goes wrong at the wrong value — "0.6" explains nothing on
its own.
"""

from __future__ import annotations

import pytest

from avatar.audio.turn_detection import (
    END_OF_TURN_SILENCE_MS,
    EventKind,
    TurnDetector,
    TurnEvent,
)

FRAME_MS = 32
SPEECH = 0.9
DIP = 0.45  # below onset, above release: mid-word
QUIET = 0.02


def detector(**overrides: object) -> TurnDetector:
    settings: dict[str, object] = {
        "frame_ms": FRAME_MS,
        "onset_frames": 3,
        "min_speech_ms": 200,
        "end_of_turn_silence_ms": 320,  # 10 frames, keeps the tests short
    }
    settings.update(overrides)
    return TurnDetector(**settings)  # type: ignore[arg-type]


def feed(det: TurnDetector, probability: float, frames: int) -> list[TurnEvent]:
    events: list[TurnEvent] = []
    for _ in range(frames):
        events.extend(det.push(probability))
    return events


def kinds(events: list[TurnEvent]) -> list[EventKind]:
    return [e.kind for e in events]


# -- onset -----------------------------------------------------------------


def test_onset_requires_consecutive_frames_above_the_bar() -> None:
    det = detector(onset_frames=3)

    assert feed(det, SPEECH, 2) == [], "two frames is not yet speech"
    events = feed(det, SPEECH, 1)

    assert kinds(events) == [EventKind.SPEECH_START]


def test_a_single_loud_frame_does_not_interrupt_the_avatar() -> None:
    """
    A cough, a door, a keyboard.

    Acting on onset interrupts the avatar mid-sentence, so a false positive is visible
    and expensive. Transients rarely sustain for three frames; speech always does.
    """
    det = detector(onset_frames=3)

    assert feed(det, SPEECH, 1) == []
    assert feed(det, QUIET, 5) == []
    assert not det.in_speech


def test_the_onset_run_must_be_unbroken() -> None:
    """
    Otherwise a noisy room accumulates its way past the threshold.

    Two loud frames, a gap, two more loud frames is not two-thirds of an utterance —
    it is noise, and counting it as progress toward onset would make the detector
    steadily more trigger-happy the noisier the room got.
    """
    det = detector(onset_frames=3)

    feed(det, SPEECH, 2)
    feed(det, QUIET, 1)
    assert feed(det, SPEECH, 2) == [], "the run restarted, so this is only two frames"
    assert kinds(feed(det, SPEECH, 1)) == [EventKind.SPEECH_START]


def test_onset_credits_the_frames_that_established_it() -> None:
    """The confirming frames were speech; the detector was only withholding judgment."""
    det = detector(onset_frames=3)

    feed(det, SPEECH, 3)

    assert det.speech_ms == 3 * FRAME_MS


# -- hysteresis ------------------------------------------------------------


def test_a_mid_word_dip_does_not_end_the_turn() -> None:
    """
    Speech probability drops inside words — plosives, the gap before a stressed
    syllable. A single threshold turns each of those into an end-of-turn, and the
    avatar starts answering halfway through the candidate's sentence.
    """
    det = detector()
    feed(det, SPEECH, 3)

    events = feed(det, DIP, 8)  # below onset, above release

    assert events == [], "the dip is still speech"
    assert det.in_speech


def test_release_bar_above_onset_is_rejected_at_construction() -> None:
    """
    Inverted hysteresis makes speech harder to sustain than to enter.

    The result is an end-of-turn inside almost every word, which presents as a wildly
    over-eager avatar and is very hard to trace back to two numbers in a config.
    """
    with pytest.raises(ValueError, match="hysteresis"):
        TurnDetector(frame_ms=FRAME_MS, onset_probability=0.4, release_probability=0.6)


# -- end of turn -----------------------------------------------------------


def test_end_of_turn_waits_for_the_full_silence_window() -> None:
    det = detector(end_of_turn_silence_ms=320)  # 10 frames
    feed(det, SPEECH, 10)

    assert feed(det, QUIET, 9) == [], "nine frames of silence is a pause, not a turn"
    events = feed(det, QUIET, 1)

    assert kinds(events) == [EventKind.END_OF_TURN]


def test_end_of_turn_reports_speech_excluding_the_trailing_silence() -> None:
    det = detector(end_of_turn_silence_ms=320)
    feed(det, SPEECH, 10)

    events = feed(det, QUIET, 10)

    assert events[0].speech_ms == 10 * FRAME_MS, "the silence is not part of the speech"


def test_speech_resuming_inside_the_window_cancels_the_end_of_turn() -> None:
    """A thinking pause. Answering into it is worse than waiting too long."""
    det = detector(end_of_turn_silence_ms=320)
    feed(det, SPEECH, 8)  # comfortably over min_speech_ms, so the end is a real turn

    feed(det, QUIET, 8)
    assert feed(det, SPEECH, 1) == [], "still the same turn"
    assert feed(det, QUIET, 9) == [], "the silence counter restarted"
    assert kinds(feed(det, QUIET, 1)) == [EventKind.END_OF_TURN]


def test_end_of_turn_latency_is_the_configured_silence_window() -> None:
    """
    This term is a policy choice, not a measurement.

    Whatever is set here appears in the perceived turnaround unchanged, and no amount
    of GPU makes it smaller — which is why PROCESS.md 1.5 names it as often the largest
    term and this test pins the relationship.
    """
    det = detector(end_of_turn_silence_ms=480)
    feed(det, SPEECH, 20)

    silent_frames = 0
    while True:
        events = det.push(QUIET)
        silent_frames += 1
        if events:
            break

    assert silent_frames * FRAME_MS == 480


# -- retraction ------------------------------------------------------------


def test_speech_shorter_than_a_syllable_is_retracted_not_delivered() -> None:
    """
    Onset can be wrong, and an empty turn handed to the LLM is worse than none.

    Retraction maps onto the orchestrator's `on_speech_retract`, which returns the
    session to IDLE without consuming a turn epoch.
    """
    det = detector(onset_frames=3, min_speech_ms=200, end_of_turn_silence_ms=320)

    feed(det, SPEECH, 4)  # 128ms, under the 200ms floor
    events = feed(det, QUIET, 10)

    assert kinds(events) == [EventKind.SPEECH_RETRACT]
    assert events[0].speech_ms == 4 * FRAME_MS


def test_speech_just_over_the_floor_is_delivered() -> None:
    det = detector(onset_frames=3, min_speech_ms=200, end_of_turn_silence_ms=320)

    feed(det, SPEECH, 7)  # 224ms
    events = feed(det, QUIET, 10)

    assert kinds(events) == [EventKind.END_OF_TURN]


# -- sequences -------------------------------------------------------------


def test_two_turns_in_a_row() -> None:
    det = detector(end_of_turn_silence_ms=320)

    first = feed(det, SPEECH, 10) + feed(det, QUIET, 10)
    second = feed(det, SPEECH, 10) + feed(det, QUIET, 10)

    assert kinds(first) == [EventKind.SPEECH_START, EventKind.END_OF_TURN]
    assert kinds(second) == [EventKind.SPEECH_START, EventKind.END_OF_TURN]


def test_a_retraction_does_not_poison_the_next_turn() -> None:
    det = detector(min_speech_ms=200, end_of_turn_silence_ms=320)

    feed(det, SPEECH, 4)
    retracted = feed(det, QUIET, 10)
    delivered = feed(det, SPEECH, 10) + feed(det, QUIET, 10)

    assert kinds(retracted) == [EventKind.SPEECH_RETRACT]
    assert kinds(delivered) == [EventKind.SPEECH_START, EventKind.END_OF_TURN]


def test_silence_alone_emits_nothing_forever() -> None:
    det = detector()

    assert feed(det, QUIET, 500) == []


def test_continuous_speech_emits_onset_once_and_then_nothing() -> None:
    det = detector()

    events = feed(det, SPEECH, 500)

    assert kinds(events) == [EventKind.SPEECH_START]


def test_reset_forgets_the_turn_without_emitting() -> None:
    """
    Called after a barge-in: the turn the detector was tracking is no longer the turn
    in progress, and emitting an end-of-turn for it would start a second turn on top of
    the one the barge-in just created.
    """
    det = detector(end_of_turn_silence_ms=320)
    feed(det, SPEECH, 10)
    assert det.in_speech

    det.reset()

    assert not det.in_speech
    assert det.speech_ms == 0
    assert feed(det, QUIET, 20) == [], "no end-of-turn for a turn we abandoned"


# -- construction ----------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"frame_ms": 0}, "frame_ms"),
        ({"frame_ms": -1}, "frame_ms"),
        ({"frame_ms": 32, "onset_frames": 0}, "onset_frames"),
    ],
)
def test_invalid_settings_are_rejected(kwargs: dict[str, int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        TurnDetector(**kwargs)  # type: ignore[arg-type]


def test_defaults_are_internally_consistent() -> None:
    det = TurnDetector(frame_ms=FRAME_MS)

    assert det.release_probability <= det.onset_probability
    assert det.end_of_turn_silence_ms == END_OF_TURN_SILENCE_MS
    assert det.min_speech_ms < det.end_of_turn_silence_ms, (
        "a floor above the silence window could never be reached"
    )
