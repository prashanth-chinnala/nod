"""
The placeholder LLM, TTS, and idle loop.

These are stand-ins, and the tests are here because the properties they stand in for
are real. `chunk_into_sentences` in particular survives M4 unchanged: the real model
adapter yields tokens, this yields speakable units, and nothing downstream notices
which model produced them. Getting the flush behaviour wrong here would show up as a
multi-second turnaround with every component performing perfectly.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from avatar.audio.tts import SAMPLE_RATE, ToneTTS, tone_pcm
from avatar.bmp import solid_bmp
from avatar.idle import load_idle_loop, placeholder_idle_loop
from avatar.llm import (
    MAX_CHUNK_CHARS,
    ScriptedInterviewer,
    chunk_into_sentences,
    split_sentences,
)
from avatar.mixer import TARGET_FPS
from avatar.png import decode as png_decode


async def stream(*tokens: str) -> AsyncIterator[str]:
    for token in tokens:
        yield token


async def collect(source: AsyncIterator[str]) -> list[str]:
    return [item async for item in source]


class InstantSleep:
    """Records requested delays without incurring them."""

    def __init__(self) -> None:
        self.requested: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.requested.append(seconds)


# -- sentence splitting ----------------------------------------------------


def test_split_keeps_terminators_with_their_sentence() -> None:
    assert split_sentences("One. Two? Three!") == ["One.", " Two?", " Three!"]


def test_split_keeps_an_unterminated_tail() -> None:
    assert split_sentences("No terminator here") == ["No terminator here"]


def test_split_of_empty_text_is_empty() -> None:
    assert split_sentences("") == []


# -- chunking --------------------------------------------------------------


async def test_chunker_emits_as_soon_as_a_sentence_closes() -> None:
    chunks = await collect(chunk_into_sentences(stream("Hello", " there.", " Next", " one.")))

    # Two units, not one: the first is speakable before the second token arrives,
    # and emitting it early is the whole point.
    assert chunks == ["Hello there.", " Next one."]


async def test_chunker_flushes_a_long_run_with_no_punctuation() -> None:
    token = "x" * 40
    tokens = [token] * 6  # 240 chars, no terminator anywhere

    chunks = await collect(chunk_into_sentences(stream(*tokens), max_chars=MAX_CHUNK_CHARS))

    # A model that never punctuates must not buffer until it finishes; that is the
    # stall this exists to prevent.
    assert len(chunks) > 1
    assert "".join(chunks) == "".join(tokens)


async def test_chunker_emits_a_trailing_partial_sentence() -> None:
    chunks = await collect(chunk_into_sentences(stream("Done.", " Unfinished")))

    assert chunks == ["Done.", " Unfinished"]


async def test_chunker_of_an_empty_stream_yields_nothing() -> None:
    assert await collect(chunk_into_sentences(stream())) == []


# -- scripted interviewer --------------------------------------------------


async def test_interviewer_yields_more_than_one_sentence_per_turn() -> None:
    sleep = InstantSleep()
    llm = ScriptedInterviewer(["First part. Second part."], ttft_ms=200, sleep=sleep)

    assert await collect(llm([])) == ["First part.", " Second part."]
    assert sleep.requested == [0.2], "time-to-first-token is modelled; total time is not"


async def test_interviewer_advances_with_the_number_of_answers_given() -> None:
    llm = ScriptedInterviewer(["Q1.", "Q2."], sleep=InstantSleep())

    first = await collect(llm([]))
    second = await collect(llm([{"role": "assistant", "content": "Q1."}]))

    assert first == ["Q1."]
    assert second == ["Q2."]


async def test_interviewer_cycles_rather_than_running_out() -> None:
    llm = ScriptedInterviewer(["Q1.", "Q2."], sleep=InstantSleep())
    history = [{"role": "assistant", "content": "x"}] * 2

    assert await collect(llm(history)) == ["Q1."]


def test_interviewer_rejects_an_empty_script() -> None:
    with pytest.raises(ValueError, match="at least one question"):
        ScriptedInterviewer([])


# -- tone TTS --------------------------------------------------------------


def test_tone_pcm_length_matches_the_requested_duration() -> None:
    pcm = tone_pcm(100, 220.0)

    assert len(pcm) == int(SAMPLE_RATE * 0.1) * 2  # mono, 16-bit


def test_tone_pcm_is_within_the_declared_amplitude() -> None:
    import struct

    pcm = tone_pcm(20, 440.0, amplitude=0.22)
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)

    assert max(abs(s) for s in samples) <= int(32767 * 0.22)


async def test_tts_chunk_durations_sum_to_the_estimated_duration() -> None:
    tts = ToneTTS(chunk_ms=80, sleep=InstantSleep())
    text = "one two three four five"  # five words -> 2000ms at 150wpm

    chunks = [c async for c in tts(text, epoch=3)]

    assert sum(c.duration_ms for c in chunks) == tts.duration_ms(text) == 2000
    assert all(c.epoch == 3 for c in chunks)


async def test_tts_never_emits_a_chunk_longer_than_the_chunk_size() -> None:
    tts = ToneTTS(chunk_ms=80, sleep=InstantSleep())

    chunks = [c async for c in tts("one two three", epoch=1)]

    # The epoch check that cancels a turn runs once per chunk, so chunk size is the
    # floor on barge-in latency through this stage.
    assert all(c.duration_ms <= 80 for c in chunks)
    assert chunks[-1].duration_ms > 0


async def test_tts_models_time_to_first_audio_before_the_first_chunk() -> None:
    sleep = InstantSleep()
    tts = ToneTTS(chunk_ms=80, first_audio_delay_ms=150, realtime_factor=1.0, sleep=sleep)

    await anext(tts("one two", epoch=1))

    assert sleep.requested == [0.15]


async def test_tts_generates_ahead_of_playback_but_not_infinitely() -> None:
    sleep = InstantSleep()
    tts = ToneTTS(chunk_ms=80, first_audio_delay_ms=0, realtime_factor=4.0, sleep=sleep)

    chunks = [c async for c in tts("one two", epoch=1)]

    played_ms = sum(c.duration_ms for c in chunks)
    generation_ms = sum(sleep.requested) * 1000
    # Yielding instantly would make first-frame latency meaningless and let the
    # mixer's queue grow without bound.
    assert 0 < generation_ms < played_ms
    assert generation_ms == pytest.approx(played_ms / 4)


def test_tone_amplitude_varies_at_a_syllable_rate() -> None:
    """
    The envelope is load-bearing, not cosmetic.

    The stub renderer derives its mouth opening from per-frame RMS. A flat carrier
    would give constant RMS and a placeholder whose mouth never moves, so the demo
    could not show audio driving video at all.
    """
    import struct

    pcm = tone_pcm(1000, 220.0)  # ~3.6 syllables at SYLLABLE_HZ
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    window = SAMPLE_RATE // 40  # 25ms
    peaks = [
        max(abs(s) for s in samples[i : i + window])
        for i in range(0, len(samples) - window, window)
    ]

    assert min(peaks) < 0.7 * max(peaks), "envelope must actually swing"


def test_tone_chunks_join_without_a_phase_discontinuity() -> None:
    """
    Restarting the phase per chunk clicks twelve times a second.

    It sounds like a transport fault and is not one, so the joint is checked here.
    """
    import struct

    whole = tone_pcm(160, 220.0)
    first = tone_pcm(80, 220.0, start_ms=0)
    second = tone_pcm(80, 220.0, start_ms=80)

    assert first + second == whole, "a slice must equal the same span of the whole"

    joined = struct.unpack(f"<{len(first + second) // 2}h", first + second)
    boundary = len(first) // 2
    step_at_joint = abs(joined[boundary] - joined[boundary - 1])
    typical = max(abs(joined[i] - joined[i - 1]) for i in range(1, min(200, len(joined))))
    assert step_at_joint <= typical * 2, "sample-to-sample step jumped at the boundary"


async def test_tts_of_empty_text_yields_nothing() -> None:
    tts = ToneTTS(sleep=InstantSleep())

    assert [c async for c in tts("   ", epoch=1)] == []


def test_tts_rejects_a_non_positive_realtime_factor() -> None:
    with pytest.raises(ValueError, match="realtime_factor"):
        ToneTTS(realtime_factor=0)


# -- idle loop -------------------------------------------------------------


def test_placeholder_loop_is_exactly_one_breath_long() -> None:
    loop = placeholder_idle_loop(width=8, height=8)

    # One full sine period, so the last frame flows into the first with no
    # cross-fade. A real clip almost never has that property.
    assert len(loop) == int(4.0 * TARGET_FPS)


def test_placeholder_loop_frames_are_decodable_images() -> None:
    """
    The idle loop shares the stub's rasteriser, so it shares its wire format.

    Was asserting BMP magic bytes; the format became PNG once 108 KB frames were
    measured at 22 Mbps. Decoding is the better assertion anyway -- it would catch a
    truncated or mis-strided frame, which a magic-byte check happily passes.
    """
    loop = placeholder_idle_loop(width=8, height=8)

    frame = loop.next_frame()

    width, height, rows = png_decode(frame.data)
    assert (width, height) == (8, 8)
    assert len(rows) == 8 and all(len(r) == 8 * 3 for r in rows)


def test_placeholder_loop_declares_every_frame_a_clean_exit() -> None:
    loop = placeholder_idle_loop(width=8, height=8)

    # True in the only sense available: a solid rectangle has no mouth to be caught
    # open. Marking a subset would look more rigorous and mean nothing.
    assert all(loop.at_clean_exit() or loop.next_frame() for _ in range(len(loop)))


def test_load_rejects_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no prepared idle loop"):
        load_idle_loop(tmp_path / "nope")


def test_load_rejects_a_directory_with_no_frames(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"no \.bmp frames"):
        load_idle_loop(tmp_path)


def test_load_rejects_frames_without_a_manifest(tmp_path: Path) -> None:
    (tmp_path / "000.bmp").write_bytes(solid_bmp(4, 4, (1, 2, 3)))

    # Defaulting to "every frame is fine" would produce an intermittent visible pop
    # at the start of some turns and not others, which is close to undebuggable from
    # a recording. The error belongs at load time.
    with pytest.raises(FileNotFoundError, match=r"mouth_closed\.json"):
        load_idle_loop(tmp_path)


def test_load_rejects_a_stale_manifest(tmp_path: Path) -> None:
    (tmp_path / "000.bmp").write_bytes(solid_bmp(4, 4, (1, 2, 3)))
    (tmp_path / "mouth_closed.json").write_text("[0, 9]")

    with pytest.raises(ValueError, match="stale"):
        load_idle_loop(tmp_path)


def test_load_reads_frames_in_sort_order(tmp_path: Path) -> None:
    for i in range(3):
        (tmp_path / f"{i:03d}.bmp").write_bytes(solid_bmp(4, 4, (i, i, i)))
    (tmp_path / "mouth_closed.json").write_text(json.dumps([0, 2]))

    loop = load_idle_loop(tmp_path)

    assert len(loop) == 3
    assert loop.at_clean_exit() is True  # index 0
    loop.next_frame()
    assert loop.at_clean_exit() is False  # index 1
