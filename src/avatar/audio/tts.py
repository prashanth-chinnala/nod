"""
Streaming speech synthesis.

`ToneTTS` is not a text-to-speech engine. It emits a sine tone of the correct
duration, chunked and paced the way a streaming TTS engine would, and it exists so
that the renderer, transport, and client can be exercised against audio with honest
timing before a real engine is wired in. It is audible, which matters: barge-in is
much easier to demonstrate on a Loom when the avatar is making a sound.

Three properties are real and are the reason a silent stub would not do:

  duration    derived from word count, so `heard_text`'s estimate and the audio the
              client actually plays describe the same utterance
  chunking    fixed-size chunks, so the renderer receives a stream and never a
              whole file -- the constraint M2's real renderer has to satisfy
  pacing      generation runs ahead of playback but not infinitely, which is how a
              real engine behaves and what keeps the mixer's buffer bounded

What is fake: the waveform. Replacing this with a real engine is a change to one
constructor argument in `server.py`, because everything above talks to
`SpeechStream`.
"""

from __future__ import annotations

import asyncio
import math
import struct
from collections.abc import AsyncGenerator

from avatar.contracts import AudioChunk, Sleep

SAMPLE_RATE = 16_000
"""Mono 16-bit PCM. Low enough to be cheap on the wire, high enough to be audible."""

BYTES_PER_SAMPLE = 2

CHUNK_MS = 80
"""
Chunk size handed downstream.

Smaller chunks mean lower time-to-first-audio and more per-chunk overhead in the
renderer; larger chunks mean the epoch check that cancels a turn runs less often, so
barge-in latency has this as its floor. 80ms is two frames at 25fps.
"""

WORDS_PER_MINUTE = 150.0
"""
Must match `orchestrator.WORDS_PER_MINUTE`.

They agree by convention rather than by construction, which is a real if small
coupling: history truncation estimates duration from word count, and if this engine
spoke at a different rate the estimate would be wrong in proportion. A real engine
resolves it properly by reporting word-level timestamps, and then neither constant
matters.
"""


SYLLABLE_HZ = 3.6
"""
Syllable rate of the amplitude envelope. Roughly conversational English.

Not cosmetic. A constant-amplitude tone has constant RMS, and the stub renderer
derives its mouth opening from exactly that -- so a flat carrier would produce a
placeholder whose mouth never moves, and the demo could not show that audio is
driving video at all. The envelope is what makes the two visibly coupled.
"""

ENVELOPE_FLOOR = 0.35
"""How quiet the envelope gets between syllables. Never zero: silence reads as a dropout."""

HARMONIC_MIX = 0.35
"""Second-harmonic weight. Takes the edge off an otherwise very pure sine."""


def tone_pcm(
    duration_ms: int,
    frequency_hz: float,
    *,
    start_ms: int = 0,
    amplitude: float = 0.22,
) -> bytes:
    """
    Mono s16le tone with a syllable-rate amplitude envelope.

    `start_ms` is this slice's offset within the utterance, and it matters: carrier
    phase and envelope phase are both computed from absolute sample position, so
    consecutive chunks join without a discontinuity. Restarting the phase per chunk
    puts an audible click at every 80ms boundary -- twelve a second -- which sounds
    like a transport fault and is not one.

    Amplitude is kept low; this plays in someone's headphones.
    """
    first = int(SAMPLE_RATE * start_ms / 1000)
    count = int(SAMPLE_RATE * duration_ms / 1000)
    if count <= 0:
        return b""

    carrier_step = 2 * math.pi * frequency_hz / SAMPLE_RATE
    envelope_step = 2 * math.pi * SYLLABLE_HZ / SAMPLE_RATE
    swing = 1.0 - ENVELOPE_FLOOR
    # Normalise so carrier + harmonic still peaks at 1.0 and amplitude means what it says.
    scale = int(32767 * amplitude / (1.0 + HARMONIC_MIX))

    values = []
    for offset in range(count):
        n = first + offset
        envelope = ENVELOPE_FLOOR + swing * (0.5 - 0.5 * math.cos(envelope_step * n))
        carrier = math.sin(carrier_step * n) + HARMONIC_MIX * math.sin(2 * carrier_step * n)
        values.append(int(scale * envelope * carrier))

    return struct.pack(f"<{count}h", *values)


class ToneTTS:
    """A `SpeechStream` with real timing and a fake voice."""

    def __init__(
        self,
        *,
        chunk_ms: int = CHUNK_MS,
        first_audio_delay_ms: int = 120,
        words_per_minute: float = WORDS_PER_MINUTE,
        realtime_factor: float = 4.0,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if realtime_factor <= 0:
            raise ValueError("realtime_factor must be positive")
        self.chunk_ms = chunk_ms
        self.first_audio_delay_ms = first_audio_delay_ms
        self.words_per_minute = words_per_minute
        self.realtime_factor = realtime_factor
        self._sleep = sleep

    def duration_ms(self, text: str) -> int:
        if not text.strip():
            return 0
        return int(len(text.split()) / self.words_per_minute * 60_000)

    def __call__(self, text: str, epoch: int) -> AsyncGenerator[AudioChunk, None]:
        return self._generate(text, epoch)

    async def _generate(self, text: str, epoch: int) -> AsyncGenerator[AudioChunk, None]:
        total_ms = self.duration_ms(text)
        if total_ms <= 0:
            return

        # Stands in for time-to-first-audio, which is a real and substantial term in
        # the latency budget and the one sentence-chunking exists to bound.
        await self._sleep(self.first_audio_delay_ms / 1000)

        # A pitch derived from the text, so successive sentences are audibly
        # distinct and a barge-in mid-sentence is obvious on a recording.
        frequency = 180.0 + (hash(text) % 7) * 40.0

        emitted_ms = 0
        while emitted_ms < total_ms:
            chunk_ms = min(self.chunk_ms, total_ms - emitted_ms)
            yield AudioChunk(
                # start_ms keeps carrier and envelope phase continuous across chunk
                # boundaries; without it every 80ms join clicks.
                pcm=tone_pcm(chunk_ms, frequency, start_ms=emitted_ms),
                epoch=epoch,
                duration_ms=chunk_ms,
            )
            emitted_ms += chunk_ms
            # Generation runs ahead of playback, but by a bounded factor. Yielding
            # the whole utterance instantly would make first-frame latency
            # meaningless and let the mixer's queue grow without limit.
            await self._sleep(chunk_ms / 1000 / self.realtime_factor)
