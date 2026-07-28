"""
The session state machine, expressed as data.

Two tables live here and nowhere else:

  LEGAL_TRANSITIONS  which state changes are permitted
  FRAME_SOURCE       which frame source each state draws from

Both are data rather than `if` chains for the same reason: a table can be
enumerated by a test, and a chain cannot. `test_state_machine.py` walks
LEGAL_TRANSITIONS exhaustively, so adding a state without deciding its
transitions and its frame source fails the suite rather than silently producing a
state the mixer has no opinion about.

This module imports only from `contracts`. No torch, no renderer, no transport.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum, auto


class State(Enum):
    INITIALIZING = auto()
    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()
    CANCELLING = auto()
    CLOSED = auto()

    def __str__(self) -> str:
        return self.name


class FrameSource(Enum):
    """Where the mixer pulls the next frame from."""

    IDLE_LOOP = auto()
    RENDERER = auto()


LEGAL_TRANSITIONS: Mapping[State, frozenset[State]] = {
    # Renderer session opened, transport track opened.
    State.INITIALIZING: frozenset({State.IDLE, State.CLOSED}),
    # THINKING direct from IDLE is the silence re-prompt: the system starts a turn
    # nobody asked for, so it never passes through LISTENING.
    State.IDLE: frozenset({State.LISTENING, State.THINKING, State.CLOSED}),
    # IDLE covers VAD retraction -- noise that looked like speech onset.
    State.LISTENING: frozenset({State.THINKING, State.IDLE, State.CLOSED}),
    # IDLE covers an empty generation or a mid-turn exception. CANCELLING covers
    # barge-in before a single frame was rendered.
    State.THINKING: frozenset({State.SPEAKING, State.IDLE, State.CANCELLING, State.CLOSED}),
    State.SPEAKING: frozenset({State.IDLE, State.CANCELLING, State.CLOSED}),
    # CANCELLING always lands in LISTENING: the only way to get here is that the
    # candidate is already speaking.
    State.CANCELLING: frozenset({State.LISTENING, State.CLOSED}),
    State.CLOSED: frozenset(),
}

FRAME_SOURCE: Mapping[State, FrameSource] = {
    State.INITIALIZING: FrameSource.IDLE_LOOP,
    State.IDLE: FrameSource.IDLE_LOOP,
    State.LISTENING: FrameSource.IDLE_LOOP,
    # No dedicated "thinking" visual. A deliberate scope choice, not an oversight:
    # a distinct thinking pose is a second idle clip plus a second seam to get
    # right, and the brief puts visual fidelity out of scope.
    State.THINKING: FrameSource.IDLE_LOOP,
    State.SPEAKING: FrameSource.RENDERER,
    # Back to the idle loop the instant cancellation starts. Waiting until
    # LISTENING would keep stale lip movement on screen for the length of the
    # cancellation, which is the visible symptom of a laggy barge-in.
    State.CANCELLING: FrameSource.IDLE_LOOP,
    State.CLOSED: FrameSource.IDLE_LOOP,
}


class InvalidTransition(RuntimeError):
    """Raised when the orchestrator attempts a transition not in the table."""

    def __init__(self, source: State, target: State) -> None:
        super().__init__(f"illegal transition {source} -> {target}")
        self.source = source
        self.target = target


def can_transition(source: State, target: State) -> bool:
    return target in LEGAL_TRANSITIONS[source]
