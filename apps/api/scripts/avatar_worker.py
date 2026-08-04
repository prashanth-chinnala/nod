#!/usr/bin/env python3
"""
Run the avatar as a LiveKit participant that publishes audio and video together.

**What this proves, and it is the point of the whole exercise.** Today audio leaves via
`transport.send_audio()` and video via the mixer's cadence loop: two publishers, two clocks,
and a
measured trailing gap of −66 ms to +172 ms with turns up to 538 ms late. This joins a room once
and pushes both media through a single `rtc.AVSynchronizer`, which pairs them. The gap stops
being something to tune and becomes something the framework maintains.

**Step 1 of `docs/LIVEKIT_AVATAR_NOTES.md`, deliberately.** One process, `QueueAudioOutput`, and
the
**stub** renderer producing raw RGB24 — so this runs with no GPU and no second process, and what
it exercises is the interface and the idle-loop handover rather than the model. `--renderer
musetalk` switches to the real one where a card exists.

    docker compose --env-file .env.development up -d      # the SFU must be running
    python scripts/avatar_worker.py --seconds 12
    python scripts/avatar_worker.py --seconds 12 --renderer musetalk --reference media/x.mp4

It reports what it published and what it produced, and exits non-zero if the room never
connected or no frame was ever paired -- because "it ran without error" is not the same claim as
"video reached a room", and only the second one is worth anything here.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from avatar.config import load_env  # noqa: E402

load_env()

from avatar.avstream import AvStream  # noqa: E402
from avatar.contracts import AudioChunk, FrameCodec, RendererConfig  # noqa: E402
from avatar.idle import placeholder_idle_loop  # noqa: E402
from avatar.orchestrator import (  # noqa: E402
    RENDER_LEAD_IN_FRAMES,
    SEAM_WAIT_MAX_MS,
)
from avatar.presentation import FramePresenter, SeamGate  # noqa: E402
from avatar.state import FrameSource  # noqa: E402
from avatar.telemetry import Telemetry  # noqa: E402
from avatar.transport.livekit_avatar import (  # noqa: E402
    LiveKitVideoGenerator,
    SegmentEpochs,
)

SAMPLE_RATE = 16_000
CHUNK_MS = 20


def speech(seconds: float, epoch: int) -> list[AudioChunk]:
    """
    Synthetic speech-shaped PCM, in the chunk size a TTS adapter emits.

    A tone rather than real speech, and that is fine here: this measures whether audio and video
    reach a room paired, which is a function of chunk sizes and timing rather than of what the
    audio says. Amplitude varies so a listener can tell it is not silence.
    """
    import math
    import struct

    per_chunk = SAMPLE_RATE * CHUNK_MS // 1000
    chunks: list[AudioChunk] = []
    for index in range(int(seconds * 1000 / CHUNK_MS)):
        samples = []
        for n in range(per_chunk):
            t = (index * per_chunk + n) / SAMPLE_RATE
            envelope = 0.35 * (1.0 + math.sin(2 * math.pi * 1.3 * t)) / 2.0
            samples.append(int(32767 * envelope * math.sin(2 * math.pi * 190.0 * t)))
        chunks.append(
            AudioChunk(
                pcm=struct.pack(f"<{per_chunk}h", *samples),
                epoch=epoch,
                duration_ms=CHUNK_MS,
            )
        )
    return chunks


async def run(args: argparse.Namespace) -> int:
    from livekit import rtc
    from livekit.agents.voice.avatar import AvatarOptions, AvatarRunner, QueueAudioOutput

    from avatar.renderers import build
    from avatar.transport.livekit import credentials, room_token

    telemetry = Telemetry()

    # Raw pixels, because `rtc.VideoFrame` takes a buffer and runs its own H.264. The renderer
    # is told rather than left to guess -- the binding refuses an encoded frame naming this
    # setting instead of decoding it back into the shape it started in.
    #
    # Two mechanisms because the two renderers have two: the stub takes a constructor argument,
    # and MuseTalk reads the environment so that `renderer_options()` -- the fixed set every
    # renderer must accept -- does not grow a key only two of them understand. Both are set here
    # so neither depends on which renderer was chosen.
    os.environ["AVATAR_FRAME_CODEC"] = FrameCodec.RGB24
    renderer_options: dict[str, object] = (
        {"codec": FrameCodec.RGB24} if args.renderer == "stub" else {}
    )
    renderer = build(RendererConfig(name=args.renderer, options=renderer_options))
    loader = getattr(renderer, "load", None)
    if callable(loader):
        loader()

    identity = renderer.prepare_identity(args.reference)
    session = renderer.start_session(identity)

    # A persona's own idle loop where the renderer has one, the placeholder where it does not.
    # `idle_loop` is `MuseTalkRenderer`'s -- it builds the loop from the reference's own
    # quietest frames, which is what makes standing by look like the same person. The stub has
    # no persona, so the breathing placeholder is not a fallback there, it is the correct
    # answer.
    supplier = getattr(renderer, "idle_loop", None)
    idle = supplier(identity) if callable(supplier) else None
    if idle is None:
        idle = placeholder_idle_loop(
            width=args.width, height=args.height, codec=FrameCodec.RGB24
        )
        print(f"-- no persona idle loop; using the placeholder at {args.width}x{args.height}")

    presenter = FramePresenter(idle, telemetry)
    stream = AvStream(presenter, frame_interval_ms=args.frame_interval_ms)

    # One epoch for the whole run. A real deployment reads the orchestrator's; see the binding's
    # `_epoch`, which has no default precisely so a split process cannot silently get this
    # wrong.
    # The epoch source, and it differs by topology on purpose.
    #
    # In-process the orchestrator owns the number; here the script does, and one turn is the
    # whole run. Split, the worker derives it from segment boundaries -- see `SegmentEpochs`,
    # which is why no turn number has to cross the wire. A single default for both would make
    # one of them silently wrong, which is the failure this codebase keeps paying for.
    epochs: Any = SegmentEpochs() if args.audio == "stream" else (lambda: 1)

    seam = SeamGate(
        presenter,
        telemetry,
        lead_in_frames=RENDER_LEAD_IN_FRAMES,
        seam_wait_max_ms=SEAM_WAIT_MAX_MS,
        clock=time.monotonic,
    )

    def render_arriving_audio(chunk: AudioChunk) -> None:
        """
        Turn received audio into rendered frames, cutting to them only on a clean seam.

        Only wired in stream mode: in queue mode this script generates the audio and drives
        the renderer itself. Without it the worker published its idle loop forever while audio
        crossed the wire perfectly -- visible only because `frames_discarded` stayed at zero.

        **`seam.maybe_cut()` rather than `set_source(RENDERER)`.** The earlier version cut the
        moment the first chunk arrived, skipping both the lead-in cushion and the mouth-closed
        check -- so the handover popped, and popped invisibly, because nothing in a count
        reveals it. The gate is the orchestrator's own policy, shared rather than rewritten.
        """
        renderer.push_audio(session, chunk)
        for frame in renderer.frames(session):
            presenter.offer(frame, epochs())
        seam.maybe_cut()
    generator = LiveKitVideoGenerator(
        stream,
        sample_rate=SAMPLE_RATE,
        epoch=epochs,
        on_audio=render_arriving_audio if args.audio == "stream" else None,
        on_segment_end=seam.turn_ended if args.audio == "stream" else None,
    )

    probe = idle_frame_shape(idle)
    url, _, _ = credentials()
    room = rtc.Room()

    # Two ways audio can arrive, which is the whole difference between step 1 and step 3.
    #
    # `queue` is in-process: this script generates audio and hands it straight over. It proves
    # the interface and the pairing with nothing else in the way.
    #
    # `stream` is the split: audio arrives over `lk.audio_stream` from another participant, and
    # a barge-in arrives as the `lk.clear_buffer` RPC. Nothing about the generator changes --
    # which is the point of having built it against our own types.
    if args.audio == "queue":
        audio_recv: Any = QueueAudioOutput(sample_rate=SAMPLE_RATE)
    else:
        from livekit.agents.voice.avatar import DataStreamAudioReceiver

        audio_recv = DataStreamAudioReceiver(room, sender_identity=args.sender)
    audio_out = audio_recv

    avatar_options = AvatarOptions(
        video_width=probe[0],
        video_height=probe[1],
        # From the same constant the server uses. A worker picking its own frame rate is the bug
        # `ea11841` fixed on the measurement side, and it would be worse here: the synchroniser
        # paces to this number, so a wrong one desyncs everything it pairs.
        video_fps=1000.0 / args.frame_interval_ms,
        audio_sample_rate=SAMPLE_RATE,
        audio_channels=1,
    )
    runner = AvatarRunner(
        room, audio_recv=audio_out, video_gen=generator, options=avatar_options
    )

    print(f"-- connecting to {url} as {args.identity!r} in room {args.room!r}")
    await room.connect(url, room_token(args.room, args.identity, name="Avatar"))
    print(f"-- connected: {room.isconnected()}")
    if not room.isconnected():
        print("!! the room never connected; is the SFU running and LIVEKIT_URL reachable?")
        return 1

    # A watcher, and it is not optional. `AvatarRunner._publish_track` awaits
    # `wait_for_subscription()` on the audio publication before it streams anything -- so with
    # nobody in the room the forward loop blocks after a single frame and the run looks like a
    # deadlock in our code. It is the opposite: the worker refuses to render to an empty room,
    # which is exactly right on a paid GPU and is the single most useful thing this script
    # discovered. Headless verification therefore has to bring its own audience.
    watcher = await watch(args.room)
    print("-- a subscriber joined; the runner will publish now")

    await runner.start()
    print(
        f"-- runner started, publishing {probe[0]}x{probe[1]} "
        f"at {avatar_options.video_fps:.1f} fps"
    )

    async def feed() -> None:
        """Push audio the way the orchestrator would: chunk by chunk, in real time."""
        await asyncio.sleep(1.0)
        print("-- speaking")
        presenter.set_source(FrameSource.RENDERER)
        for chunk in speech(args.speak_seconds, epochs()):
            renderer.push_audio(session, chunk)
            for frame in renderer.frames(session):
                presenter.offer(frame, epochs())
            await audio_out.capture_frame(
                rtc.AudioFrame(
                    data=chunk.pcm,
                    sample_rate=SAMPLE_RATE,
                    num_channels=1,
                    samples_per_channel=len(chunk.pcm) // 2,
                )
            )
            await asyncio.sleep(CHUNK_MS / 1000.0)
        audio_out.flush()
        print("-- done speaking; back to the idle loop")
        presenter.set_source(FrameSource.IDLE_LOOP)

    feeder = (
        asyncio.create_task(feed())
        if args.audio == "queue"
        else asyncio.create_task(asyncio.sleep(args.seconds))
    )
    if args.audio == "stream":
        print(f"-- waiting for audio on lk.audio_stream from {args.sender or 'any agent'}")
    started = time.monotonic()
    try:
        while time.monotonic() - started < args.seconds:
            await asyncio.sleep(0.5)
    finally:
        feeder.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await feeder
        published = [
            publication.sid
            for publication in room.local_participant.track_publications.values()
        ]
        await runner.aclose()
        await room.disconnect()
        await watcher.room.disconnect()
        renderer.close_session(session)

    print("\n-- what happened")
    print(f"   tracks published         {len(published)}")
    print(f"   video frames emitted     {stream.frames_emitted}")
    print(f"   audio chunks emitted     {stream.audio_emitted}")
    print(f"   frames repeated          {presenter.frames_repeated}")
    print(f"   epoch at exit            {epochs()}")
    print(f"   seams forced             {seam.seams_forced}")
    print(f"   frames discarded         {presenter.frames_discarded}")
    print(f"   video frames RECEIVED    {watcher.video} by a remote subscriber")
    print(f"   audio frames RECEIVED    {watcher.audio} by a remote subscriber")
    median, worst, final = watcher.drift()
    print(f"   audio media decoded      {watcher.audio_media_s:.2f}s")
    print(f"   video media span         {watcher.video_span_s:.2f}s")
    print(
        f"   A/V DRIFT at subscriber  median {median:+.0f} ms, worst {worst:+.0f} ms, "
        f"final {final:+.0f} ms"
    )

    # Asserted on what arrived, not on what was sent. "We published" is a claim about our own
    # process; "a subscriber decoded 240 frames" is a claim about the system.
    if len(published) < 2:
        print("\n!! fewer than two tracks published; both media should be on the room")
        return 1
    if watcher.video == 0 or watcher.audio == 0:
        print("\n!! a subscriber received no paired media, so nothing was actually proven")
        return 1
    print(
        f"\n   {watcher.video} video and {watcher.audio} audio frames reached a remote "
        "subscriber, from one participant through one synchroniser."
    )
    return 0


class Watcher:
    """
    A second participant that subscribes, counts what arrives, and measures the drift.

    The verification this script exists for. Counting what we *published* measures our own
    process; counting what a remote subscriber *decoded* measures the system, including the SFU,
    the negotiated codec and the synchroniser's pacing. Only the second is worth reporting.

    **The drift is the point, and it is measurable here without epochs.** A subscriber sees
    decoded media, not turns, so per-turn attribution is impossible from this side -- but the
    two media timelines are both visible: video frames carry `timestamp_us` and audio frames
    carry `duration`.
    Comparing how much of each has arrived is exactly the quantity `AVSynchronizer` exists to
    hold together, and it is the claim the whole migration rests on. MEASUREMENTS §8b measured
    the same thing over WebSocket at −66 ms to +172 ms with turns up to 538 ms late.
    """

    def __init__(self, room: object) -> None:
        self.room = room
        self.video = 0
        self.audio = 0
        self.audio_media_s = 0.0
        """Playback seconds of audio decoded, accumulated from each frame's own duration."""
        self.video_span_s = 0.0
        """Span of video media timestamps, first frame to last."""
        self._first_video_us: int | None = None
        self._last_video_us: int | None = None
        self.samples: list[tuple[float, float, float]] = []
        """(wall, audio_media_s, video_media_s) per video frame, for the drift curve."""
        self._t0: float | None = None
        self._tasks: list[asyncio.Task[None]] = []

    def attach(self, track: object) -> None:
        from livekit import rtc

        if isinstance(track, rtc.RemoteVideoTrack):
            self._tasks.append(asyncio.create_task(self._video(rtc.VideoStream(track))))
        elif isinstance(track, rtc.RemoteAudioTrack):
            self._tasks.append(asyncio.create_task(self._audio(rtc.AudioStream(track))))

    def _mark(self) -> float:
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        return now - self._t0

    async def _video(self, stream: object) -> None:
        with contextlib.suppress(Exception):
            async for event in stream:  # type: ignore[attr-defined]
                self.video += 1
                stamp = getattr(event, "timestamp_us", None)
                if stamp is not None:
                    if self._first_video_us is None:
                        self._first_video_us = stamp
                    self._last_video_us = stamp
                    self.video_span_s = (stamp - self._first_video_us) / 1_000_000
                self.samples.append((self._mark(), self.audio_media_s, self.video_span_s))

    async def _audio(self, stream: object) -> None:
        with contextlib.suppress(Exception):
            async for event in stream:  # type: ignore[attr-defined]
                self.audio += 1
                # `duration` is the frame's own playback length, so this accumulates the audio
                # timeline from the media rather than from wall clock -- the two differ by
                # exactly the thing being measured.
                self.audio_media_s += float(getattr(event.frame, "duration", 0.0))

    def drift(self) -> tuple[float, float, float]:
        """
        (median, worst, final) drift in milliseconds, video media minus audio media.

        Positive means video is ahead of the audio it should be paired with. Sampled on video
        frames because those are the sparser stream, and the first second is dropped: both
        tracks are still being negotiated there and the numbers describe the handshake rather
        than the steady state.
        """
        settled = [s for s in self.samples if s[0] > 1.0 and s[1] > 0.0]
        if not settled:
            return (0.0, 0.0, 0.0)
        deltas = [(video - audio) * 1000 for _, audio, video in settled]
        ordered = sorted(deltas)
        median = ordered[len(ordered) // 2]
        worst = max(deltas, key=abs)
        return (median, worst, deltas[-1])


async def watch(room_name: str) -> Watcher:
    """Join as a subscriber, so the worker has an audience and something counts the result."""
    from livekit import rtc

    from avatar.transport.livekit import credentials, room_token

    url, _, _ = credentials()
    room = rtc.Room()
    watcher = Watcher(room)

    @room.on("track_subscribed")
    def _on_subscribed(track: object, *_: object) -> None:
        watcher.attach(track)

    await room.connect(url, room_token(room_name, "avatar-watcher", name="Watcher"))
    return watcher


def idle_frame_shape(idle: object) -> tuple[int, int]:
    """
    The published track's dimensions, taken from an actual idle frame.

    Read rather than configured: the synchroniser is told a width and height once, and a track
    negotiated at the wrong size shows as a stretched or cropped face for the whole session. The
    frame itself is the only thing that knows.
    """
    frame = idle.next_frame()  # type: ignore[attr-defined]
    if not frame.is_raw:
        raise SystemExit(
            f"this path needs raw pixels and the renderer produced {frame.codec!r}. Pass "
            "codec=rgb24 to the renderer -- decoding here would undo an encode it should not "
            "have paid for."
        )
    return frame.width, frame.height


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", default="avatar-worker-check")
    parser.add_argument("--identity", default="avatar-agent")
    parser.add_argument("--renderer", default="stub", choices=("stub", "musetalk"))
    parser.add_argument("--reference", default="reference.mp4")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--speak-seconds", type=float, default=4.0)
    parser.add_argument("--frame-interval-ms", type=int, default=40)
    parser.add_argument(
        "--audio",
        default="queue",
        choices=("queue", "stream"),
        help="queue: in-process (step 1). stream: audio over lk.audio_stream (step 3)",
    )
    parser.add_argument(
        "--sender",
        default="",
        help="identity publishing audio in stream mode; empty means the first agent",
    )
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
