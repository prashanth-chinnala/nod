"""
LiveKit as a `Transport`: the agent joins the room as a participant and publishes tracks.

**Why this exists.** `transport/websocket.py` documents what the shortcut gives up, and the list
is real: TCP head-of-line blocking turns one lost packet into a stall for every frame behind it,
there is no congestion feedback to adapt to, and the jitter buffer is 150ms hand-rolled in the
browser. An SFU fixes all three by being the thing that was designed to.

**What it buys beyond robustness, and this is the larger point.** Recording stops being
something to build. Egress is a room configuration rather than a service that muxes audio and
video server-side and manages storage — so "the session produces a reviewable artifact" becomes
infrastructure rather than code. Nothing in this file does recording; that is the point.

**What it does not buy: speed.** A full turn measures 2.7-5.8s and the three dominant terms are
the 700ms turn policy, LLM time-to-first-token, and TTS. Transport is roughly 20ms of that on
loopback. WebRTC is the correct production shape and a robustness win; anyone expecting it to
close the latency gap will be disappointed, and the numbers in PROCESS.md must be re-measured
over a real link rather than carried across.

**The orchestrator does not change.** It receives a `Transport` and has never known whether that
is a WebSocket, an SFU, or a test double. This is the second implementation of that Protocol,
which is what makes the boundary a claim rather than an assertion.

**One asymmetry worth stating.** The WebSocket transport also carried telemetry and control
messages. Here those travel over LiveKit's data channel, which is a separate path from the media
tracks — so a congested video track no longer delays a barge-in signal. That is an improvement,
and it is also a behaviour change: the two are no longer ordered relative to each other.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from typing import Any

from avatar.contracts import AudioChunk, Frame

SAMPLE_RATE = 16_000
NUM_CHANNELS = 1

DEFAULT_URL = "ws://localhost:7880"
"""
Where the SFU is. A local dev server by default so a clean clone can run one.

LiveKit Cloud is the same code path with a different URL and key -- nothing below this constant
knows the difference, which is why self-hosted and cloud are a configuration choice rather than
a fork.
"""

AGENT_IDENTITY = "avatar-agent"
"""
The agent's participant identity, fixed rather than random.

The browser subscribes to *this* identity's tracks. A random one would work only because there
happens to be one other participant, and would break the moment a second human joined -- an
observer sitting in on an interview is a feature someone will ask for.
"""


def credentials() -> tuple[str, str, str]:
    """URL, API key, and secret. Raises with the variable names rather than half-connecting."""
    url = os.environ.get("LIVEKIT_URL", DEFAULT_URL)
    key = os.environ.get("LIVEKIT_API_KEY", "")
    secret = os.environ.get("LIVEKIT_API_SECRET", "")
    if not key or not secret:
        raise RuntimeError(
            "the livekit transport needs LIVEKIT_API_KEY and LIVEKIT_API_SECRET. "
            "For a local dev server both are printed by `livekit-server --dev`."
        )
    return url, key, secret


def room_token(room: str, identity: str, *, name: str = "") -> str:
    """
    A join token for one participant in one room.

    Scoped to a single room by name, not a wildcard grant. The candidate's link already is not a
    credential (see the console's session page); a token that admitted the bearer to *any* room
    would make that much worse, and scoping it costs one line.
    """
    from livekit import api

    _, key, secret = credentials()
    grants = api.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True)
    return (
        api.AccessToken(key, secret)
        .with_identity(identity)
        .with_name(name or identity)
        .with_grants(grants)
        .to_jwt()
    )


class LiveKitTransport:
    """
    Publishes the avatar's audio and video into a room, and control messages over data.

    Satisfies `Transport` structurally, like `WebSocketTransport` does. Conformance is asserted
    in the test suite rather than by inheritance, so a second transport cannot subclass its way
    to compliance while quietly taking different arguments.
    """

    def __init__(
        self,
        room_name: str,
        *,
        width: int,
        height: int,
        identity: str = AGENT_IDENTITY,
    ) -> None:
        self.room_name = room_name
        self.width = width
        self.height = height
        self.identity = identity
        self._room: Any = None
        self._audio_source: Any = None
        self._video_source: Any = None
        self._connected = False
        self._marked_epoch = -1
        """
        The last epoch announced on the data channel.

        `-1` rather than `0`, because epoch 0 is a real turn -- the opening question -- and
        starting at 0 would swallow its marker and leave the first turn of every session
        unmeasurable, which is exactly the turn a demo looks at.
        """
        self.recording: dict[str, Any] = {
            "status": "off",
            "reason": "the session has not started",
        }
        """
        What happened when recording was set up, filled in by `connect`.

        Public and plain data, because the server stores it on the session record and the
        console renders it. An attribute rather than a callback: the transport already knows the
        answer by the time anything else could ask.
        """

    async def connect(self) -> None:
        """
        Create the room, then join it.

        Separate from `open_track` because joining can fail for reasons that have nothing to do
        with the track -- a wrong URL, an expired token -- and those deserve a different error
        than "the avatar has no video".

        The room is created explicitly before the join, and the order is load-bearing: LiveKit
        auto-creates a room for the first participant to arrive, and an auto-created room
        carries no egress configuration. Join first and the interview records nothing, with no
        error anywhere. `ensure_room` is idempotent and returns a status rather than raising, so
        a deployment with no egress service still holds the interview -- the outcome is kept on
        `recording` for the session record to store, because a recording that was never set up
        must be visible on the record rather than discovered when someone asks for the video.
        """
        from livekit import rtc

        from avatar.transport.recording import ensure_room

        self.recording = await ensure_room(self.room_name)

        url, _, _ = credentials()
        self._room = rtc.Room()
        await self._room.connect(url, room_token(self.room_name, self.identity))
        self._connected = True

    async def open_track(self) -> None:
        """
        Publish one audio and one video track, and keep them for the whole session.

        Published once at session start rather than per turn, for the same reason the WebSocket
        track is continuous: a track that only exists while speaking stalls between turns, and a
        stall is more visible than a dropped frame -- it also corrupts the receiver's jitter
        estimate, so the recovery is worse than the original gap.
        """
        from livekit import rtc

        if not self._connected:
            await self.connect()

        self._audio_source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
        self._video_source = rtc.VideoSource(self.width, self.height)

        audio = rtc.LocalAudioTrack.create_audio_track("avatar-voice", self._audio_source)
        video = rtc.LocalVideoTrack.create_video_track("avatar-face", self._video_source)

        await self._room.local_participant.publish_track(audio)
        await self._room.local_participant.publish_track(video)

    async def send_audio(self, chunk: AudioChunk) -> None:
        """
        Push PCM into the audio source.

        No jitter buffer here, deliberately: WebRTC owns that now. The 150ms cushion the browser
        client hand-rolled existed because a raw WebSocket has no such machinery, and keeping
        both would add its latency on top of the one that actually adapts to the link.
        """
        if self._audio_source is None:
            return
        from livekit import rtc

        samples = len(chunk.pcm) // 2
        if not samples:
            return
        frame = rtc.AudioFrame(
            data=chunk.pcm,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            samples_per_channel=samples,
        )
        await self._audio_source.capture_frame(frame)

    async def send_frame(self, frame: Frame) -> None:
        """
        Push one rendered frame into the video source.

        The frame arrives encoded -- PNG from the placeholder, JPEG from a photographic model --
        because that is what the WebSocket wire wanted. WebRTC wants raw pixels and does its own
        encoding, so this decodes first. That is genuinely wasted work: encode, decode, then
        encode again in VP8. The right fix is for the renderer to hand over raw frames and let
        the transport decide, which means a `Frame` variant carrying pixels -- deferred rather
        than pretended away, and it is a real cost of having built the WebSocket path first.

        **The epoch goes out separately, over the data channel.** A WebSocket frame is a framed
        binary message the client can read an epoch out of; a WebRTC video track is just pixels,
        so the turn a frame belongs to is erased in transit. That is why first-frame and
        end-to-end latency were unmeasurable over WebRTC: the client could see a frame arrive
        and had no way to say which turn it closed. Marking the first frame of each epoch on the
        data channel restores the attribution without touching the media path.

        Sent before the frame is captured, and only on an epoch change -- one message per turn,
        not per frame at 25fps. The two travel different paths, so ordering is not guaranteed;
        what the client can then measure is stated precisely where it measures it, in
        `lib/rtc.ts`.
        """
        if self._video_source is None:
            return
        from livekit import rtc

        if frame.epoch != self._marked_epoch:
            self._marked_epoch = frame.epoch
            await self.send_control({"type": "frame_epoch", "epoch": frame.epoch})

        rgba, width, height = _decode_to_rgba(frame.data, self.width, self.height)
        await asyncio.to_thread(
            self._video_source.capture_frame,
            rtc.VideoFrame(width, height, rtc.VideoBufferType.RGBA, rgba),
        )

    async def flush_audio(self) -> None:
        """
        Barge-in. Clears whatever the SFU has not yet delivered.

        `clear_queue` is the SFU-side half. The browser must still stop its own playback, which
        is why the control message goes out too -- a server-only flush leaves the candidate
        hearing a sentence the avatar abandoned, which reads as a laggy interruption no matter
        how fast the state machine reacted.
        """
        if self._audio_source is not None:
            self._audio_source.clear_queue()
        await self.send_control({"type": "flush_audio"})

    async def send_control(self, payload: dict[str, Any]) -> None:
        """
        Control and telemetry over the data channel, which is a separate path from the media.

        Reliable rather than lossy: a dropped `flush_audio` would leave the client playing
        abandoned speech, and a dropped state change would leave the UI describing a state the
        session is no longer in. Neither is worth the latency saving.
        """
        if self._room is None:
            return

        await self._room.local_participant.publish_data(
            json.dumps(payload, default=str).encode(), reliable=True, topic="control"
        )

    async def close_track(self) -> None:
        """Leave the room. Tolerant of a connection already gone, because teardown races it."""
        if self._room is not None:
            # Teardown races the connection dropping, so a disconnect that fails because the
            # room is already gone is the expected path, not an error.
            with contextlib.suppress(Exception):
                await self._room.disconnect()
        self._room = None
        self._audio_source = None
        self._video_source = None
        self._connected = False


def _decode_to_rgba(data: bytes, width: int, height: int) -> tuple[bytes, int, int]:
    """
    Encoded image bytes to raw RGBA.

    Pillow rather than a hand-rolled decoder: the frames are PNG or JPEG, and this is exactly
    the boring dependency that should be used rather than reimplemented. It is imported here so
    a process that never uses this transport does not pay for it.
    """
    import io

    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        converted = image.convert("RGBA")
        return converted.tobytes(), converted.width, converted.height
