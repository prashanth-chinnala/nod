"""
A renderer that needs no GPU, no weights, and no network.

This exists for two reasons, and the second one matters more than the first:

  1. CI can exercise the entire session machine -- lifecycle, barge-in, cadence,
     history truncation -- with nothing installed but pytest.
  2. It is the proof that `TalkingHeadRenderer` is a real boundary. A second
     implementation that shares no code with the first, swapped by a one-line
     config change, is evidence; a Protocol with one implementation is a claim.

Frames are solid-colour 24-bit BMPs, cycling through a palette so that a stalled
track is visually distinguishable from a running one during the browser demo. BMP
because it encodes in twenty lines of pure Python and browsers render it -- no
image library, so nothing leaks into the CI dependency set.

The stub also models the one behaviour that dominates first-frame latency in real
talking-head models: they need a lookahead window of audio before they can emit
anything. `first_frame_delay_ms` is that window, and it is configurable so the
mixer's lead-in buffer can be tested against a slow renderer without owning one.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from avatar.bmp import solid_bmp
from avatar.contracts import IDLE_EPOCH, AudioChunk, Frame

FRAME_INTERVAL_MS = 40  # 25fps

PALETTE: tuple[tuple[int, int, int], ...] = (
    (40, 58, 68),
    (46, 66, 78),
    (52, 74, 88),
    (46, 66, 78),
)


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
    _colour_index: int = field(default=0, repr=False)


class StubRenderer:
    """
    Solid-colour frames paced off pushed audio.

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
        self._cache: dict[tuple[int, int, int], bytes] = {}

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
        # Propagating the chunk's epoch onto the frames it produces is the whole
        # of the renderer's involvement in cancellation. It never decides anything.
        state.epoch = chunk.epoch

    def frames(self, session: object) -> Iterator[Frame]:
        state = _as_session(session)
        available = max(0, state.audio_ms - self.first_frame_delay_ms)
        due = available // self.frame_interval_ms
        while state.frames_emitted < due:
            colour = PALETTE[state._colour_index % len(PALETTE)]
            state._colour_index += 1
            yield Frame(
                data=self._bmp(colour),
                epoch=state.epoch,
                # The mixer restamps this; the renderer's own pts is only useful
                # for spotting a renderer that has lost count of its own output.
                pts_ms=state.frames_emitted * self.frame_interval_ms,
            )
            state.frames_emitted += 1

    def reset(self, session: object) -> None:
        """Drop queued audio and frame position. Safe with nothing in flight."""
        state = _as_session(session)
        state.audio_ms = 0
        state.frames_emitted = 0
        state.epoch = IDLE_EPOCH
        state.resets += 1

    def close_session(self, session: object) -> None:
        state = _as_session(session)
        state.closed = True
        self.sessions_closed += 1

    # -- internals ----------------------------------------------------------

    def _bmp(self, colour: tuple[int, int, int]) -> bytes:
        cached = self._cache.get(colour)
        if cached is None:
            cached = solid_bmp(self.width, self.height, colour)
            self._cache[colour] = cached
        return cached


def _as_session(session: object) -> StubSession:
    if not isinstance(session, StubSession):
        raise TypeError(f"expected StubSession, got {type(session).__name__}")
    if session.closed:
        raise RuntimeError("render session is closed")
    return session
