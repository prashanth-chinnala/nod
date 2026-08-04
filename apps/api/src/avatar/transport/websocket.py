"""
Chunked frames and audio over a WebSocket.

This is the documented shortcut the brief permits in place of WebRTC. What it gives
up, stated plainly because §3.4 of PROCESS.md has to account for it:

  - No jitter buffer. TCP head-of-line blocking turns a lost packet into a stall
    for every frame behind it, where WebRTC would have dropped one frame and
    carried on. On a good connection this is invisible; on a bad one it is the
    whole difference.
  - No congestion control feedback. Nothing tells the renderer to shed resolution
    when the link degrades, so the degradation ladder in §1.6 has no input.
  - No browser-native decode path. Each frame is decoded in JS rather than by the
    platform's video decoder, which costs main-thread time and rules out hardware
    decode.
  - Audio and video are interleaved on one ordered channel, so they cannot drift
    apart -- which is convenient here and is exactly the property WebRTC gives up
    in exchange for being able to drop late video without stalling audio.

What it buys: it works, in a day, with no SFU and no ICE. M7 revisits it.

Deliberately knows nothing about FastAPI, Starlette, or any web framework. It takes
two send callables, so the codec and the `Transport` implementation are testable
without installing a web stack -- which is what keeps the CI dependency set lean.
"""

from __future__ import annotations

import struct
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import IntEnum

from avatar.contracts import AudioChunk, Frame

SendBytes = Callable[[bytes], Awaitable[None]]
SendText = Callable[[str], Awaitable[None]]


class Kind(IntEnum):
    VIDEO = 1
    AUDIO = 2


HEADER = struct.Struct("!BIII")
"""kind, pts_ms, epoch, payload length. 13 bytes."""

HEADER_SIZE = HEADER.size

MAX_PAYLOAD = 8 << 20
"""
Refuse to frame anything larger than 8MiB.

A single frame that big means an encoder misconfiguration, and sending it would
block the socket for long enough to stall the whole track. Failing loudly beats a
mysterious freeze.
"""


@dataclass(frozen=True, slots=True)
class Envelope:
    """A decoded wire message."""

    kind: Kind
    pts_ms: int
    epoch: int
    payload: bytes


def encode(kind: Kind, pts_ms: int, epoch: int, payload: bytes) -> bytes:
    """
    Pack one message.

    The length prefix is redundant over a WebSocket, which already delimits
    messages. It is here so the same codec works over a raw stream -- and because
    a client that computes the payload length independently catches a truncated
    message rather than rendering garbage.
    """
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload of {len(payload)} bytes exceeds {MAX_PAYLOAD}")
    if pts_ms < 0 or epoch < 0:
        raise ValueError(f"pts_ms and epoch must be non-negative, got {pts_ms}, {epoch}")
    return HEADER.pack(int(kind), pts_ms, epoch, len(payload)) + payload


def decode(message: bytes) -> Envelope:
    """Unpack one message. Raises ValueError on anything malformed."""
    if len(message) < HEADER_SIZE:
        raise ValueError(f"message of {len(message)} bytes is shorter than the header")
    raw_kind, pts_ms, epoch, length = HEADER.unpack_from(message)
    body = message[HEADER_SIZE:]
    if len(body) != length:
        raise ValueError(f"header declares {length} payload bytes, found {len(body)}")
    try:
        kind = Kind(raw_kind)
    except ValueError as exc:
        raise ValueError(f"unknown message kind {raw_kind}") from exc
    return Envelope(kind=kind, pts_ms=pts_ms, epoch=epoch, payload=body)


def encode_frame(frame: Frame) -> bytes:
    return encode(Kind.VIDEO, frame.pts_ms, frame.epoch, frame.data)


def encode_audio(chunk: AudioChunk, pts_ms: int) -> bytes:
    return encode(Kind.AUDIO, pts_ms, chunk.epoch, chunk.pcm)


class WebSocketTransport:
    """
    `Transport` over a pair of send callables.

    `flush_audio` sends a control message rather than dropping anything locally,
    and that is the load-bearing detail: by the time a barge-in happens, the audio
    is already in the client's Web Audio graph. A server-side flush alone leaves
    the candidate listening to a sentence the avatar abandoned, which reads as a
    laggy interruption even though the state machine reacted in microseconds.
    """

    def __init__(self, send_bytes: SendBytes, send_text: SendText) -> None:
        self._send_bytes = send_bytes
        self._send_text = send_text
        self._audio_pts_ms = 0
        self.open = False
        self.bytes_sent = 0

    async def open_track(self) -> None:
        self.open = True
        await self._send_text('{"type":"track_open"}')

    async def send_audio(self, chunk: AudioChunk) -> None:
        message = encode_audio(chunk, self._audio_pts_ms)
        self._audio_pts_ms += chunk.duration_ms
        self.bytes_sent += len(message)
        await self._send_bytes(message)

    async def send_frame(self, frame: Frame) -> None:
        message = encode_frame(frame)
        self.bytes_sent += len(message)
        await self._send_bytes(message)

    async def flush_audio(self) -> None:
        # The client stops the scheduled buffer sources and reports back how much
        # it had actually played, which is what history truncation runs on.
        await self._send_text('{"type":"flush_audio"}')

    async def close_track(self) -> None:
        self.open = False
        await self._send_text('{"type":"track_close"}')

    def end_of_turn(self) -> None:
        """
        Nothing to send, and adding a message would be worse than this no-op.

        The client learns the utterance ended by its audio queue draining, which it already
        tracks to decide when to re-enable the microphone. A `turn_end` message would be a second
        source of truth for the same fact, and the two would disagree the moment one arrived
        before the audio it describes had finished playing.
        """
