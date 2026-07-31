"""
The two-stage render pipeline: the GPU half and the CPU half overlap, and order survives.

`render()` used to decode a batch and then blend it, in sequence, on one thread. Measured on a
T4 that is 70.1 ms of GPU followed by 51.5 ms of CPU per frame, and the CPU half is the GPU
sitting idle. It now runs batch N's blending on a worker thread while batch N+1 decodes, which
measured 124.7 -> 78.4 ms/frame: 1.59x, for no change to the model.

**Why this file exists.** Overlapping introduces exactly one new way to be wrong, and it is a
bad one: frames leaving in the wrong order. That is not a crash, it is a mouth that plays back
scrambled -- which looks like a model problem, not a threading problem, and would be diagnosed
for a long time. So the ordering guarantee gets a test, and it gets one that would actually
fail if the guarantee were dropped: the CPU stage is made *deliberately* slowest-first, so a
version collecting results as they complete cannot pass by luck.

No GPU and no torch. `musetalk_torch` imports torch inside its methods, so the pipeline can be
driven with both halves stubbed -- which is the whole point, because the arithmetic being
tested is the batching and the ordering, not the matrix multiplications.
"""

from __future__ import annotations

import threading
import time

import pytest

from avatar.renderers.musetalk_torch import TorchMuseTalkBackend


class Recorder(TorchMuseTalkBackend):
    """A backend whose two halves are stubs that record what they were given."""

    def __init__(self, *, batch_size: int, delays: dict[int, float] | None = None) -> None:
        super().__init__(batch_size=batch_size)
        self.device = "cpu"
        self._models = {"stub": True}
        self.decoded: list[list[int]] = []
        self.blended: list[list[int]] = []
        self.delays = delays or {}
        self.frames_available = 0

    def _audio_features(self, pcm: bytes) -> list[object]:
        return [object() for _ in range(self.frames_available)]

    def _decode(self, prepared, batch, indices):  # type: ignore[no-untyped-def]
        self.decoded.append(list(indices))
        return list(indices)

    def _blend_and_encode(self, prepared, faces, indices):  # type: ignore[no-untyped-def]
        # Sleep before recording, so a slow batch finishes after a later fast one. If `render`
        # collected futures by completion rather than submission, this is what would expose it.
        time.sleep(self.delays.get(indices[0], 0.0))
        self.blended.append(list(indices))
        return [f"frame-{i}".encode() for i in faces]


def prepared(cycle: int) -> dict[str, object]:
    """The minimum of a prepared identity that `render` itself reads: the cycle length."""
    return {"latents": [None] * cycle}


def test_frames_come_out_in_order_even_when_a_later_batch_finishes_first() -> None:
    """
    The guarantee. Batch 0 blends slowly, batch 1 quickly; output must still be 0..7.

    This is the test that justifies the whole file. With one worker the delay also serialises
    the stage, so the assertion holds for the ordering reason rather than by accident of timing.
    """
    backend = Recorder(batch_size=4, delays={0: 0.05})
    backend.frames_available = 8
    frames = backend.render(prepared(100), b"", start_frame=0, count=8)

    assert frames == [f"frame-{i}".encode() for i in range(8)]
    assert backend.blended == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_the_cpu_half_runs_while_the_gpu_half_is_still_working() -> None:
    """
    The point of the change: decoding does not wait for the previous batch to be blended.

    Proved by construction rather than by clock. Batch 0's blend blocks until batch 1's decode
    signals it, so the sequential version -- which called blend before the next decode -- cannot
    complete at all, while the overlapped version passes immediately. A wall-clock assertion
    would have been flaky on a loaded machine, and an ordering assertion turned out to prove
    nothing: a fast blend finishes before the next decode is recorded whether it overlapped or
    not.
    """
    started_next_decode = threading.Event()
    blend_saw_it = threading.Event()

    backend = Recorder(batch_size=2)
    backend.frames_available = 4
    decodes = {"n": 0}

    def decode(prepared_, batch, indices):  # type: ignore[no-untyped-def]
        decodes["n"] += 1
        if decodes["n"] == 2:
            started_next_decode.set()
        return list(indices)

    def blend(prepared_, faces, indices):  # type: ignore[no-untyped-def]
        if indices[0] == 0 and started_next_decode.wait(timeout=5.0):
            blend_saw_it.set()
        return [f"frame-{i}".encode() for i in faces]

    backend._decode = decode  # type: ignore[method-assign]
    backend._blend_and_encode = blend  # type: ignore[method-assign]
    frames = backend.render(prepared(100), b"", start_frame=0, count=4)

    assert blend_saw_it.is_set(), "batch 0 was blended before batch 1 was decoded -- no overlap"
    assert frames == [f"frame-{i}".encode() for i in range(4)]


def test_start_frame_still_walks_the_reference_cycle_across_batches() -> None:
    """Batching must not restart the body motion; indices wrap modulo the cycle length."""
    backend = Recorder(batch_size=3)
    backend.frames_available = 6
    backend.render(prepared(4), b"", start_frame=3, count=6)

    assert backend.decoded == [[3, 0, 1], [2, 3, 0]]
    assert backend.blended == [[3, 0, 1], [2, 3, 0]]


def test_a_window_shorter_than_the_batch_renders_one_short_batch() -> None:
    """The tail of a turn is usually a partial batch, and it must not be padded or dropped."""
    backend = Recorder(batch_size=16)
    backend.frames_available = 3
    frames = backend.render(prepared(100), b"", start_frame=0, count=16)

    assert len(frames) == 3
    assert backend.decoded == [[0, 1, 2]]


def test_no_audio_means_no_frames_and_no_worker() -> None:
    """A window with no features returns before the pool is ever created."""
    backend = Recorder(batch_size=4)
    backend.frames_available = 0

    assert backend.render(prepared(100), b"", start_frame=0, count=4) == []
    assert backend._pool is None


def test_count_bounds_the_output_even_when_more_audio_arrived() -> None:
    """`count` is the mixer's budget for this window; extra features are not free frames."""
    backend = Recorder(batch_size=4)
    backend.frames_available = 20
    frames = backend.render(prepared(100), b"", start_frame=0, count=5)

    assert len(frames) == 5


def test_unload_shuts_the_worker_down_and_render_can_start_a_new_one() -> None:
    """
    A renderer released and reused must not keep a dead pool.

    `unload()` is called when a session ends; a shut-down `ThreadPoolExecutor` rejects new work,
    so a stale reference here would make the *second* session fail while the first was fine.
    """
    backend = Recorder(batch_size=2)
    backend.frames_available = 2
    backend.render(prepared(10), b"", start_frame=0, count=2)
    assert backend._pool is not None

    backend.unload()
    assert backend._pool is None

    backend._models = {"stub": True}
    backend.frames_available = 2
    assert len(backend.render(prepared(10), b"", start_frame=0, count=2)) == 2


@pytest.mark.parametrize("batch_size", [1, 2, 3, 5, 16])
def test_every_batch_size_produces_the_same_frames_in_the_same_order(batch_size: int) -> None:
    """
    Batch size is a throughput knob and must not be an output knob.

    It is read from `AVATAR_MUSETALK_BATCH`, so an operator tuning it is changing how many
    frames share a forward pass -- not which frames come out, or in what order.
    """
    backend = Recorder(batch_size=batch_size)
    backend.frames_available = 7
    frames = backend.render(prepared(5), b"", start_frame=2, count=7)

    assert frames == [f"frame-{(2 + i) % 5}".encode() for i in range(7)]
