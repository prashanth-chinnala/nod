"""
The runtime half of worker delivery: stream audio to the renderer instead of rendering.

**What changes when this is in the path.** Normally the runtime renders frames itself and
publishes them. Here it renders nothing: it joins the session's room, streams synthesised speech
to the avatar worker over `lk.audio_stream`, and the worker publishes both media as its own
participant. The browser is unaffected — it already subscribes to `avatar-agent`, which is now a
different process.

**Why this is a transport and not a mode inside the orchestrator.** The orchestrator's contract
is "hand audio to a transport and offer frames to a mixer". Both still happen; the audio goes
somewhere else and the frames go nowhere. That keeps the state machine, the turn policy and
epoch cancellation untouched, which is the same reason `PairedDelivery` is a decorator — and it
is what makes this switchable per deployment rather than a fork of the pipeline.

**Identity does not travel, and that is the design.** A pooled worker did not prepare the
persona it is about to render, and the obvious fix -- serialise the prepared identity and ship
it -- means serialising roughly a gigabyte of tensors with a format, a version and an
invalidation rule. It is not needed. The worker resolves the session to a face to a **reference
path on shared storage**, and prepares from that, caching process-wide. The expensive artifact
is rebuilt once per worker rather than moved, which is the same shape as the epoch problem: the
thing that has to cross the boundary is much smaller than it first appears.

**What it gives up, stated.** Barge-in becomes an RPC rather than an integer write, and the
worker has to be there. `available()` reports the second one before a session starts rather than
after a candidate has joined.
"""

from __future__ import annotations

import os
from typing import Any

from avatar.contracts import AudioChunk, Frame
from avatar.transport.livekit import AGENT_IDENTITY

WORKER_IDENTITY_ENV = "AVATAR_WORKER_IDENTITY"
DEFAULT_WORKER_IDENTITY = AGENT_IDENTITY
"""
Who the runtime addresses, and it is not arbitrary.

The session layer hands the browser an `agent_identity` and the client subscribes to exactly
that. So the worker must claim it, which means **the runtime must not** -- two participants
cannot share an identity, and the failure mode is the SFU evicting one of them mid-interview.
That is why this transport joins under its own name and publishes no tracks at all.

Aliased rather than repeated. The string was written out in three modules -- here, the SFU
transport, and the session layer's join response -- and three copies of a value whose entire
purpose is that two processes agree on it is the setup for a bug that presents as a black video
element with nothing logged anywhere. The import costs nothing: it is a plain string in a module
whose SDK imports are all deferred.
"""


def runtime_identity(room_name: str) -> str:
    """
    What the runtime joins a session's room as. Derived, so both processes compute the same
    answer.

    The worker has to name this participant rather than wait for whoever turns up:
    `DataStreamAudioReceiver` with no `sender_identity` waits for an **agent-kind** participant,
    and the runtime is an ordinary one. Left unset, the worker connects, publishes nothing, and
    blocks until the room drops -- which is what it did, with `room disconnected while waiting
    for participant` as the only clue that the two sides disagreed about a string.

    A function rather than a constant because it depends on the room, and shared rather than
    formatted twice for the same reason `AGENT_IDENTITY` is: the failure when the two drift is
    silence with no error.
    """
    return f"runtime-{room_name}"


class WorkerAudioTransport:
    """
    Sends audio to an avatar worker over a LiveKit data stream. Publishes no media itself.

    Satisfies the same `Transport` shape the orchestrator uses. `send_frame` is a deliberate no-
    op: frames still arrive from the local renderer if one is configured, and dropping them here
    is correct rather than wasteful -- see the note on that method.
    """

    def __init__(
        self,
        room_name: str,
        *,
        identity: str = "",
        worker_identity: str = "",
        sample_rate: int = 16_000,
    ) -> None:
        self.room_name = room_name
        self.identity = identity or runtime_identity(room_name)
        self.worker_identity = worker_identity or os.environ.get(
            WORKER_IDENTITY_ENV, DEFAULT_WORKER_IDENTITY
        )
        self.sample_rate = sample_rate
        self._room: Any = None
        self._out: Any = None
        self.audio_sent = 0
        self.frames_dropped = 0

    # -- lifecycle ----------------------------------------------------------

    async def open_track(self) -> None:
        """
        Join the room and open the audio stream to the worker.

        `wait_remote_track` is passed to `DataStreamAudioOutput` and it is not optional: LiveKit
        drops a byte stream announced to a participant with no handler for the topic, and the
        worker registers its handler as part of starting. A sender that begins the moment the
        worker *joins* is too early -- which cost an hour to diagnose, because both sides log
        nothing.
        """
        from livekit import rtc
        from livekit.agents.voice.avatar import DataStreamAudioOutput

        from avatar.transport.livekit import credentials, room_token

        url, _, _ = credentials()
        self._room = rtc.Room()
        await self._room.connect(url, room_token(self.room_name, self.identity, name="Runtime"))
        self._out = DataStreamAudioOutput(
            self._room,
            destination_identity=self.worker_identity,
            sample_rate=self.sample_rate,
            wait_remote_track=rtc.TrackKind.KIND_AUDIO,
        )

    async def close_track(self) -> None:
        if self._out is not None:
            self._out.flush()
            self._out = None
        if self._room is not None:
            await self._room.disconnect()
            self._room = None

    # -- the orchestrator's surface ------------------------------------------

    async def send_audio(self, chunk: AudioChunk) -> None:
        from livekit import rtc

        if self._out is None:
            return
        await self._out.capture_frame(
            rtc.AudioFrame(
                data=chunk.pcm,
                sample_rate=self.sample_rate,
                num_channels=1,
                samples_per_channel=len(chunk.pcm) // 2,
            )
        )
        self.audio_sent += 1

    async def send_frame(self, frame: Frame) -> None:
        """
        Dropped, and counted.

        The worker renders. If a local renderer is also configured its frames arrive here and
        are discarded -- which is why they are counted rather than silently ignored: a non-zero
        count means a GPU somewhere is doing work nobody will see, and that is worth noticing in
        `/config` rather than discovering on a bill.
        """
        self.frames_dropped += 1

    async def flush_audio(self) -> None:
        """
        Barge-in. `clear_buffer` is an RPC to the worker rather than an integer write.

        The worker's own `clear_buffer` drops its queued audio *and* advances its epoch, so
        frames already rendered for the abandoned turn go stale by the check that has always
        handled cancellation. The round trip is the cost of the split; it was measured, and the
        frames do stop.
        """
        if self._out is not None:
            self._out.clear_buffer()

    # -- readiness ----------------------------------------------------------

    def end_of_turn(self) -> None:
        """
        Close the byte stream, which is how the worker learns a turn ended.

        `flush()` on `DataStreamAudioOutput` closes the stream, and the receiver surfaces that
        as `AudioSegmentEnd` -- which is the signal the worker counts to derive its epoch and to
        hand back to the idle loop. Without this the worker never learns the utterance finished:
        it holds the last rendered frame and the next turn's audio joins the previous one's
        stream.
        """
        if self._out is not None:
            self._out.flush()

    @staticmethod
    def available() -> str:
        """
        Empty if worker delivery is usable, otherwise the reason. Checked before a session
        starts.

        A reason rather than a bool, and checked early, because the alternative is a candidate
        joining a room where nothing ever publishes a face. There is no way to detect that from
        the browser except by waiting.
        """
        try:
            import livekit.agents.voice.avatar  # noqa: F401
        except ModuleNotFoundError:
            return (
                "worker delivery needs livekit-agents, which is in the [worker] extra "
                "rather than [server] so that apps/api keeps no required dependency on "
                "it. pip install -e '.[worker]'"
            )
        if not os.environ.get("LIVEKIT_API_KEY") or not os.environ.get("LIVEKIT_API_SECRET"):
            return "worker delivery needs LIVEKIT_API_KEY and LIVEKIT_API_SECRET."
        return ""
