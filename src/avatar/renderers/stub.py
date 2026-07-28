"""
A renderer that needs no GPU, no weights, and no network.

This exists for two reasons, and the second one matters more than the first:

  1. CI can exercise the entire session machine -- lifecycle, barge-in, cadence,
     history truncation -- with nothing installed but pytest.
  2. It is the proof that `TalkingHeadRenderer` is a real boundary. A second
     implementation that shares no code with the first, swapped by a one-line
     config change, is evidence; a Protocol with one implementation is a claim.

It draws five rectangles: a ground, a head, two eyes, and a mouth whose height is
derived from the actual amplitude of the audio for that frame's 40ms slice. That is
not decoration. Two properties of a real talking-head model are genuinely reproduced:

  - it consumes a fixed slice of audio per frame and produces one frame from it, so
    a caller that feeds it whole files instead of chunks breaks immediately
  - the output visibly tracks the input, so "is audio driving video, and is it in
    sync?" is answerable by watching, which is what M3's browser demo needs

Nobody will mistake five rectangles for a face. That is deliberate: an obviously
synthetic placeholder cannot be quietly confused for a working model in a demo, and
a solid colour block -- which is what this drew first -- was so nearly invisible
against the page background that a working pipeline looked broken.

It also models the behaviour that dominates first-frame latency in real models: they
need a lookahead window of audio before emitting anything. `first_frame_delay_ms` is
that window, configurable so the mixer's lead-in buffer can be tested against a slow
renderer without owning one.
"""

from __future__ import annotations

import math
from array import array
from collections.abc import Iterator
from dataclasses import dataclass, field

from avatar.bmp import RGB, Canvas
from avatar.contracts import IDLE_EPOCH, AudioChunk, Frame

FRAME_INTERVAL_MS = 40  # 25fps

MOUTH_LEVELS = 12
"""
Distinct mouth openings.

Frames are cached per level, so the rasteriser runs a dozen times per session rather
than 25 times a second. Twelve is enough that speech reads as continuous motion; more
would be invisible at this size and would only grow the cache.
"""

REFERENCE_AMPLITUDE = 0.28
"""
Input RMS treated as a fully open mouth.

Set a little above `ToneTTS`'s 0.22 peak amplitude so a normal placeholder utterance
uses most of the range without pinning at the top. A real renderer would normalise
against the reference clip instead of a constant.
"""

GROUND: RGB = (18, 27, 34)
HEAD: RGB = (86, 116, 134)
EYE: RGB = (24, 34, 42)
MOUTH: RGB = (224, 164, 88)
"""Amber, matching the 'speaking' state colour in the client and the mockup."""


def _rms(pcm: bytes) -> float:
    """Root-mean-square of signed 16-bit little-endian mono samples, normalised."""
    if len(pcm) < 2:
        return 0.0
    samples = array("h")
    # Trailing odd byte would raise; a partial sample cannot affect the result.
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not samples:
        return 0.0
    total = 0
    for sample in samples:
        total += sample * sample
    return math.sqrt(total / len(samples)) / 32768.0


def mouth_level(pcm: bytes) -> int:
    """Quantise a frame's worth of audio to a mouth opening."""
    scaled = _rms(pcm) / REFERENCE_AMPLITUDE
    return max(0, min(MOUTH_LEVELS - 1, round(scaled * (MOUTH_LEVELS - 1))))


def draw_placeholder(width: int, height: int, level: int, *, brightness: float = 1.0) -> bytes:
    """
    Five rectangles. `level` is the mouth opening, 0 (closed) to MOUTH_LEVELS - 1.

    Proportional to the frame size so the same function serves a 4x4 test frame and a
    256x144 demo frame without a special case.

    `brightness` scales the head only, and exists for the idle loop: a placeholder
    that is perfectly static is indistinguishable from a stalled track, which is the
    one failure the mixer exists to prevent and therefore the one a demo must be able
    to show is not happening.
    """
    canvas = Canvas(width, height, GROUND)

    head_w = max(1, round(width * 0.42))
    head_h = max(1, round(height * 0.74))
    head_x = (width - head_w) // 2
    head_y = round(height * 0.16)
    head = (
        HEAD
        if brightness == 1.0
        else tuple(max(0, min(255, round(c * brightness))) for c in HEAD)
    )
    canvas.fill_rect(head_x, head_y, head_w, head_h, head)  # type: ignore[arg-type]

    eye_w = max(1, round(head_w * 0.16))
    eye_h = max(1, round(head_h * 0.1))
    eye_y = head_y + round(head_h * 0.28)
    canvas.fill_rect(head_x + round(head_w * 0.22), eye_y, eye_w, eye_h, EYE)
    canvas.fill_rect(head_x + round(head_w * 0.62), eye_y, eye_w, eye_h, EYE)

    # Grows downward from a fixed upper lip, the way a jaw does.
    mouth_w = max(1, round(head_w * 0.4))
    closed_h = max(1, round(head_h * 0.035))
    open_h = max(closed_h, round(head_h * 0.26))
    mouth_h = closed_h + round((open_h - closed_h) * level / (MOUTH_LEVELS - 1))
    canvas.fill_rect(
        head_x + (head_w - mouth_w) // 2,
        head_y + round(head_h * 0.62),
        mouth_w,
        mouth_h,
        MOUTH,
    )

    return canvas.to_bmp()


@dataclass(frozen=True, slots=True)
class StubIdentity:
    reference_path: str


@dataclass
class StubSession:
    identity: StubIdentity
    audio_ms: int = 0
    frames_emitted: int = 0
    epoch: int = IDLE_EPOCH
    closed: bool = False
    resets: int = 0
    bytes_per_frame: int = 0
    pcm: bytearray = field(default_factory=bytearray, repr=False)


class StubRenderer:
    """
    An audio-driven placeholder.

    Satisfies `TalkingHeadRenderer` structurally; the conformance check lives in
    `tests/test_boundaries.py` rather than in an inheritance relationship, so the
    real renderer is never tempted to subclass its way to compliance.
    """

    def __init__(
        self,
        *,
        width: int = 64,
        height: int = 64,
        first_frame_delay_ms: int = 0,
        frame_interval_ms: int = FRAME_INTERVAL_MS,
    ) -> None:
        self.width = width
        self.height = height
        self.first_frame_delay_ms = first_frame_delay_ms
        self.frame_interval_ms = frame_interval_ms
        self.identities_prepared = 0
        self.sessions_opened = 0
        self.sessions_closed = 0
        self._cache: dict[int, bytes] = {}

    # -- contract -----------------------------------------------------------

    def prepare_identity(self, reference_path: str) -> object:
        self.identities_prepared += 1
        return StubIdentity(reference_path=reference_path)

    def start_session(self, identity: object) -> object:
        if not isinstance(identity, StubIdentity):
            raise TypeError(f"expected StubIdentity, got {type(identity).__name__}")
        self.sessions_opened += 1
        return StubSession(identity=identity)

    def push_audio(self, session: object, chunk: AudioChunk) -> None:
        state = _as_session(session)
        state.audio_ms += chunk.duration_ms
        state.pcm.extend(chunk.pcm)
        # Propagating the chunk's epoch onto the frames it produces is the whole of
        # the renderer's involvement in cancellation. It never decides anything.
        state.epoch = chunk.epoch
        if state.bytes_per_frame == 0 and chunk.duration_ms > 0 and chunk.pcm:
            # Inferred rather than configured: AudioChunk carries a duration and a
            # payload, which is enough, and a mismatched constant here would
            # desynchronise the mouth from the audio in a way that looks like a
            # model bug.
            bytes_per_ms = len(chunk.pcm) / chunk.duration_ms
            state.bytes_per_frame = max(2, int(bytes_per_ms * self.frame_interval_ms))

    def frames(self, session: object) -> Iterator[Frame]:
        state = _as_session(session)
        available = max(0, state.audio_ms - self.first_frame_delay_ms)
        due = available // self.frame_interval_ms
        while state.frames_emitted < due:
            slice_len = state.bytes_per_frame
            if slice_len and len(state.pcm) >= slice_len:
                chunk = bytes(state.pcm[:slice_len])
                del state.pcm[:slice_len]
            else:
                # Frame is due but its audio has been consumed already. A silent
                # frame is the honest output: the mouth closes.
                chunk = b""
            yield Frame(
                data=self._render(mouth_level(chunk)),
                epoch=state.epoch,
                # The mixer restamps this; the renderer's own pts is only useful for
                # spotting a renderer that has lost count of its own output.
                pts_ms=state.frames_emitted * self.frame_interval_ms,
            )
            state.frames_emitted += 1

    def reset(self, session: object) -> None:
        """Drop queued audio and frame position. Safe with nothing in flight."""
        state = _as_session(session)
        state.audio_ms = 0
        state.frames_emitted = 0
        state.epoch = IDLE_EPOCH
        state.pcm.clear()
        state.resets += 1

    def close_session(self, session: object) -> None:
        state = _as_session(session)
        state.closed = True
        state.pcm.clear()
        self.sessions_closed += 1

    # -- internals ----------------------------------------------------------

    def _render(self, level: int) -> bytes:
        cached = self._cache.get(level)
        if cached is None:
            cached = draw_placeholder(self.width, self.height, level)
            self._cache[level] = cached
        return cached


def _as_session(session: object) -> StubSession:
    if not isinstance(session, StubSession):
        raise TypeError(f"expected StubSession, got {type(session).__name__}")
    if session.closed:
        raise RuntimeError("render session is closed")
    return session
