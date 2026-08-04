"""
The transcriber surviving a dropped socket, which is the failure that made an interview deaf.

**Why this file exists.** `DeepgramSTT` had no tests. It is the component that decides whether
the interviewer can hear, and the thing that went wrong with it was not subtle: the socket
dropped once and was never reopened, so a session transcribed its first two turns and none of
the remaining thirty-eight -- including one of 10,080 ms of speech. The VAD detected that speech
correctly. The transcriber was simply gone, and nothing anywhere said so.

Every test here drives the real class through its injected `connect` hook, so what is exercised
is the reconnection and keepalive logic rather than a mock of it. Nothing here talks to
Deepgram.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from avatar.audio.stt import DeepgramSTT


class FakeSocket:
    """
    A websocket that can be told to fail, and that records what it was sent.

    `fail_after` counts *successful* sends before the socket starts refusing, which is how a
    real connection dies: it works, and then it does not.
    """

    def __init__(self, *, fail_after: int | None = None) -> None:
        self.sent: list[Any] = []
        self.fail_after = fail_after
        self.closed = False
        self._inbox: asyncio.Queue[str] = asyncio.Queue()

    async def send(self, payload: Any) -> None:
        if self.fail_after is not None and len(self.sent) >= self.fail_after:
            raise ConnectionResetError("socket is gone")
        self.sent.append(payload)

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self) -> FakeSocket:
        return self

    async def __anext__(self) -> str:
        # Blocks forever unless a test pushes a message, which matches a quiet connection and
        # lets the reader task exist without doing anything.
        return await self._inbox.get()

    def deliver(self, text: str, *, is_final: bool = True) -> None:
        self._inbox.put_nowait(
            json.dumps(
                {
                    "type": "Results",
                    "is_final": is_final,
                    "channel": {"alternatives": [{"transcript": text}]},
                }
            )
        )


def build(**kwargs: Any) -> tuple[DeepgramSTT, list[FakeSocket]]:
    """
    A transcriber whose every connection is a fresh `FakeSocket`, and the list of them.

    The list is the assertion surface: a reconnect means a second entry, and the original bug is
    exactly the absence of one.
    """
    sockets: list[FakeSocket] = []
    factories = kwargs.pop("factories", None)

    async def connect(_url: str) -> FakeSocket:
        socket = factories.pop(0)() if factories else FakeSocket()
        sockets.append(socket)
        return socket

    stt = DeepgramSTT(
        connect=connect,
        keep_alive_interval=kwargs.pop("keep_alive_interval", 0.02),
        reconnect_base_delay=kwargs.pop("reconnect_base_delay", 0.001),
        **kwargs,
    )
    return stt, sockets


async def settle(seconds: float = 0.2) -> None:
    """Let background tasks run. Real sleeps, because the code under test uses real ones."""
    await asyncio.sleep(seconds)


# -- the bug ---------------------------------------------------------------


async def test_a_dropped_socket_is_reopened_so_later_turns_still_transcribe() -> None:
    """
    The regression, stated as the thing a candidate experiences.

    Before this, one failed send made the interviewer deaf for the rest of the interview: turns
    kept happening, each carrying `[Nms of speech, no transcript]`, and the interviewer went on
    asking plausible questions of someone it could not hear.
    """
    stt, sockets = build(factories=[lambda: FakeSocket(fail_after=2)])
    await stt.connect()

    for _ in range(4):
        await stt.push_audio(b"\x00\x00" * 160)
    await settle()

    assert len(sockets) == 2, "the dropped socket was never replaced"
    assert stt.connected
    assert stt.reconnects == 1

    await stt.push_audio(b"\x01\x00" * 160)
    assert sockets[1].sent, "audio after the reconnect went nowhere"

    await stt.aclose()


async def test_audio_continues_to_be_accepted_while_the_reconnect_is_in_flight() -> None:
    """
    `push_audio` must never wait for a handshake, because the VAD is behind it on the same loop.

    That was the original justification for not reconnecting at all. It is preserved by doing
    the handshake in a task: frames during the gap are dropped, which costs a word or two of one
    turn rather than every word of the rest of the session.
    """
    stt, _ = build(factories=[lambda: FakeSocket(fail_after=0)], reconnect_base_delay=5.0)
    await stt.connect()

    # A reconnect is now pending behind a 5 s backoff. These must all return immediately.
    for _ in range(50):
        await asyncio.wait_for(stt.push_audio(b"\x00\x00" * 160), timeout=0.05)

    await stt.aclose()


async def test_a_failed_connect_at_session_start_is_retried() -> None:
    """
    "Unreachable right now" and "unreachable for the whole interview" are different facts.

    The old code treated them as the same by never trying again, so a transcriber that missed
    its one chance at session start stayed missing.
    """
    attempts = 0

    async def connect(_url: str) -> FakeSocket:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("deepgram unreachable")
        return FakeSocket()

    stt = DeepgramSTT(connect=connect, keep_alive_interval=0.02, reconnect_base_delay=0.001)
    await stt.connect()

    assert not stt.connected, "precondition: the first attempt failed"

    await settle()

    assert stt.connected, "nothing retried the connection"
    await stt.aclose()


async def test_reconnection_gives_up_rather_than_retrying_forever() -> None:
    """
    An unauthorised or unreachable Deepgram will not recover inside one interview.

    A task retrying it forever keeps a dead session warm and hides the failure in a log nobody
    reads. `reconnects` stays at zero, which is the signal that this happened.
    """
    async def connect(_url: str) -> FakeSocket:
        raise OSError("nope")

    stt = DeepgramSTT(
        connect=connect,
        keep_alive_interval=0.02,
        reconnect_base_delay=0.001,
        max_reconnect_attempts=3,
    )
    await stt.connect()
    await settle(0.3)

    assert not stt.connected
    assert stt.reconnects == 0
    await stt.aclose()


async def test_only_one_reconnect_runs_at_a_time() -> None:
    """
    Many frames arrive while the socket is down, and each one asks for a reconnect.

    Without the guard that is one handshake per frame -- fifty concurrent TLS connections to
    Deepgram for one dropped socket, which is worse than the outage.
    """
    stt, sockets = build(
        factories=[lambda: FakeSocket(fail_after=0)], reconnect_base_delay=0.05
    )
    await stt.connect()

    for _ in range(50):
        await stt.push_audio(b"\x00\x00" * 160)
    await settle(0.3)

    assert len(sockets) == 2, f"expected one replacement, got {len(sockets) - 1}"
    await stt.aclose()


# -- the keepalive, which prevents the drop rather than recovering from it --


async def test_keepalive_is_sent_while_the_candidate_is_silent() -> None:
    """
    Deepgram closes an idle stream after about ten seconds, and this goes idle every turn.

    The socket was not dying of network trouble. It was dying of politeness, every time the
    avatar spoke for longer than ten seconds -- and reconnecting after the fact still loses the
    opening words of the reply.
    """
    stt, sockets = build(keep_alive_interval=0.02)
    await stt.connect()
    await settle(0.15)

    assert stt.keep_alives > 0
    assert {"type": "KeepAlive"} in [
        json.loads(m) for m in sockets[0].sent if isinstance(m, str)
    ]
    await stt.aclose()


async def test_keepalive_stays_quiet_while_audio_is_flowing() -> None:
    """
    A keepalive during speech is pure noise on the wire, and it would confuse the read loop.

    Audio is itself proof the connection is alive, so the timer resets on every frame.
    """
    stt, sockets = build(keep_alive_interval=0.05)
    await stt.connect()

    for _ in range(30):
        await stt.push_audio(b"\x00\x00" * 160)
        await asyncio.sleep(0.005)

    texts = [m for m in sockets[0].sent if isinstance(m, str)]
    assert texts == [], f"keepalive fired during speech: {texts}"
    await stt.aclose()


# -- teardown --------------------------------------------------------------


async def test_closing_cancels_both_background_tasks() -> None:
    """
    A reconnector that outlives `aclose()` reopens what is being torn down.

    That leaves a live Deepgram stream behind every finished interview -- billed, attached to
    nothing, and invisible until the bill.
    """
    stt, sockets = build(
        factories=[lambda: FakeSocket(fail_after=0)], reconnect_base_delay=0.05
    )
    await stt.connect()
    await stt.push_audio(b"\x00\x00" * 160)

    await stt.aclose()
    before = len(sockets)
    await settle(0.3)

    assert len(sockets) == before, "a background task reopened the socket after aclose()"
    assert not stt.connected


async def test_closing_finalises_the_stream_before_dropping_it() -> None:
    """`CloseStream` asks Deepgram to flush a final rather than losing the last words."""
    stt, sockets = build()
    await stt.connect()
    await stt.aclose()

    assert {"type": "CloseStream"} in [
        json.loads(m) for m in sockets[0].sent if isinstance(m, str)
    ]
    assert sockets[0].closed
