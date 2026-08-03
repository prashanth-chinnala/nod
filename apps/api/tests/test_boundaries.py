"""
Mechanical enforcement of the module boundary.

"The ML model is one bounded, swappable piece of the system" is a claim, and this
file is the part that makes it checkable. Two things are asserted:

  1. The orchestration layer's import graph contains no ML dependency and no
     renderer implementation. This is what keeps CI GPU-free -- not a convention
     that erodes the first time someone needs a tensor.

  2. A second, unrelated renderer satisfies the same contract and is selected by a
     one-line config change. A Protocol with one implementation proves nothing.

Enforced by parsing the source rather than by importing and catching failures: the
point is that the dependency is absent from the graph, not merely that it happened
not to be installed on the machine running the suite.
"""

from __future__ import annotations

import ast
import inspect
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from avatar import contracts, mixer, orchestrator, presentation, state, telemetry
from avatar.audio.tts import SAMPLE_RATE
from avatar.contracts import RendererConfig, TalkingHeadRenderer
from avatar.png import decode as png_decode
from avatar.renderers import build
from avatar.renderers.stub import MOUTH_LEVELS, StubRenderer, draw_placeholder, mouth_level

ORCHESTRATION_MODULES = (contracts, state, telemetry, presentation, mixer, orchestrator)

FORBIDDEN_ROOTS = frozenset(
    {
        "torch",
        "torchvision",
        "torchaudio",
        "numpy",
        "cv2",
        "PIL",
        "tensorrt",
        "onnxruntime",
        "transformers",
        "diffusers",
        "librosa",
        "scipy",
    }
)

RENDERER_CONTRACT_METHODS = (
    "prepare_identity",
    "start_session",
    "push_audio",
    "frames",
    "reset",
    "close_session",
)


def imported_roots(module_path: Path) -> set[str]:
    """Top-level package names imported by a module, from its AST."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def imported_avatar_modules(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("avatar"):
            assert node.module is not None
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(a.name for a in node.names if a.name.startswith("avatar"))
    return found


@pytest.mark.parametrize("module", ORCHESTRATION_MODULES, ids=lambda m: m.__name__)
def test_orchestration_layer_has_no_ml_dependency(module: object) -> None:
    path = Path(inspect.getfile(module))  # type: ignore[arg-type]
    offenders = imported_roots(path) & FORBIDDEN_ROOTS

    assert not offenders, (
        f"{path.name} imports {sorted(offenders)}. Adding an ML dependency here "
        "breaks the GPU-free CI run; move the code behind TalkingHeadRenderer instead."
    )


@pytest.mark.parametrize("module", ORCHESTRATION_MODULES, ids=lambda m: m.__name__)
def test_orchestration_layer_does_not_import_a_renderer(module: object) -> None:
    path = Path(inspect.getfile(module))  # type: ignore[arg-type]
    offenders = {m for m in imported_avatar_modules(path) if m.startswith("avatar.renderers")}

    assert not offenders, (
        f"{path.name} imports {sorted(offenders)}. The orchestrator receives a "
        "renderer; it must never name one."
    )


def test_contracts_imports_nothing_from_the_package() -> None:
    """
    `contracts` is the root of the dependency graph.

    If it ever imports from elsewhere in the package, some other module has become
    a de facto contract and the direction of dependency has inverted.
    """
    path = Path(inspect.getfile(contracts))

    assert imported_avatar_modules(path) == set()


def test_importing_the_orchestration_layer_pulls_in_no_ml_package() -> None:
    """
    Belt to the AST braces: nothing arrives transitively either.

    **Run in a clean subprocess, and that is the whole point.** This used to read the current
    process's `sys.modules`, which conflates "importing avatar loaded this" with "something,
    anywhere in this test session, loaded this". It passed for a long time because nothing else
    in the suite touched a forbidden package, and then a test that uploads a PDF pulled in
    `pypdf`,
    which pulls in `PIL` -- and this failed with the message "importing avatar loaded ['PIL']",
    which was not true. A guard whose failure message can be false is a guard that will be
    argued with rather than fixed.

    A subprocess makes the claim literally testable: import exactly the orchestration layer into
    a fresh interpreter, and report what came with it. That is also strictly stronger, because
    it can no longer be fooled -- or falsely tripped -- by test ordering.
    """
    program = (
        "import sys, json;"
        "import avatar.orchestrator, avatar.mixer, avatar.presentation, avatar.state,"
        " avatar.contracts;"
        f"print(json.dumps(sorted(set(sys.modules) & {set(FORBIDDEN_ROOTS)!r})))"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=Path(inspect.getfile(contracts)).parents[2],
        timeout=120,
    )

    assert result.returncode == 0, f"the import itself failed:\n{result.stderr[-2000:]}"
    offenders = json.loads(result.stdout.strip() or "[]")
    assert not offenders, (
        f"importing the orchestration layer loaded {offenders}. The boundary exists so the "
        "whole suite can run with no GPU, no weights and no network."
    )


# -- the contract is real --------------------------------------------------


def test_stub_renderer_satisfies_the_contract() -> None:
    assert isinstance(StubRenderer(), TalkingHeadRenderer)


@pytest.mark.parametrize("method_name", RENDERER_CONTRACT_METHODS)
def test_stub_renderer_matches_the_contract_signature(method_name: str) -> None:
    """
    `runtime_checkable` only checks that the attribute exists.

    Parameter names are checked here so that a renderer cannot satisfy the
    Protocol at runtime while quietly taking different arguments -- mypy catches
    that in CI, and this catches it for anyone reading the test suite instead.
    """
    expected = inspect.signature(getattr(TalkingHeadRenderer, method_name))
    actual = inspect.signature(getattr(StubRenderer, method_name))

    assert list(expected.parameters) == list(actual.parameters)


def test_renderer_is_selected_by_config_alone() -> None:
    """Swapping the ML model is a change to this value and nothing else."""
    assert isinstance(build(RendererConfig(name="stub")), StubRenderer)
    assert isinstance(build(), StubRenderer)


def test_renderer_options_reach_the_implementation() -> None:
    renderer = build(RendererConfig(name="stub", options={"width": 128, "height": 96}))

    assert isinstance(renderer, StubRenderer)
    assert (renderer.width, renderer.height) == (128, 96)


def test_unknown_renderer_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unknown renderer"):
        build(RendererConfig(name="a-model-that-does-not-exist"))


# -- the stub behaves like a renderer --------------------------------------


def test_stub_emits_frames_paced_off_pushed_audio() -> None:
    from avatar.contracts import AudioChunk

    renderer = StubRenderer(width=4, height=4, frame_interval_ms=40)
    session = renderer.start_session(renderer.prepare_identity("ref.mp4"))

    assert list(renderer.frames(session)) == []

    renderer.push_audio(session, AudioChunk(pcm=b"", epoch=7, duration_ms=120))
    frames = list(renderer.frames(session))

    assert len(frames) == 3
    assert all(f.epoch == 7 for f in frames), "the renderer propagates the turn tag"
    # Decodable, not merely non-empty. The wire format changed from BMP to PNG when
    # 108 KB frames turned out to be 22 Mbps, so this asserts the payload survives a
    # real decode rather than pinning magic bytes a future renderer may not share.
    for frame in frames:
        width, height, rows = png_decode(frame.data)
        assert (width, height) == (4, 4)
        assert len(rows) == 4


def test_stub_withholds_frames_until_its_lookahead_window_is_filled() -> None:
    from avatar.contracts import AudioChunk

    renderer = StubRenderer(width=4, height=4, first_frame_delay_ms=200, frame_interval_ms=40)
    session = renderer.start_session(renderer.prepare_identity("ref.mp4"))

    renderer.push_audio(session, AudioChunk(pcm=b"", epoch=1, duration_ms=160))
    assert list(renderer.frames(session)) == []

    renderer.push_audio(session, AudioChunk(pcm=b"", epoch=1, duration_ms=80))
    assert len(list(renderer.frames(session))) == 1


def pcm_at_rms(duration_ms: int, rms: float) -> bytes:
    """PCM whose RMS is exactly `rms`. A constant sample value is the simplest way."""
    count = int(SAMPLE_RATE * duration_ms / 1000)
    value = int(32768 * rms)
    return struct.pack(f"<{count}h", *([value] * count))


def test_stub_mouth_tracks_the_amplitude_of_each_frames_audio() -> None:
    """
    The placeholder is audio-driven, not a colour cycle.

    This is what makes "is audio driving video, and is it in sync?" answerable by
    watching the demo, which is the job the browser page has to do for the Loom.
    """
    assert mouth_level(pcm_at_rms(40, 0.0)) == 0, "silence closes the mouth"
    assert mouth_level(pcm_at_rms(40, 0.30)) == MOUTH_LEVELS - 1, "loud opens it fully"

    quiet = mouth_level(pcm_at_rms(40, 0.05))
    mid = mouth_level(pcm_at_rms(40, 0.15))
    assert 0 < quiet < mid < MOUTH_LEVELS - 1, "and it is monotonic in between"


def test_stub_frames_differ_when_the_audio_differs() -> None:
    from avatar.contracts import AudioChunk

    renderer = StubRenderer(width=32, height=32, frame_interval_ms=40)
    session = renderer.start_session(renderer.prepare_identity("ref.mp4"))

    renderer.push_audio(session, AudioChunk(pcm=pcm_at_rms(40, 0.0), epoch=1, duration_ms=40))
    silent = next(renderer.frames(session)).data
    renderer.push_audio(session, AudioChunk(pcm=pcm_at_rms(40, 0.28), epoch=1, duration_ms=40))
    loud = next(renderer.frames(session)).data

    assert silent != loud, "identical frames for different audio proves nothing"


def test_stub_stretches_available_samples_across_the_declared_duration() -> None:
    """
    `duration_ms` is the timeline authority, not the payload length.

    A chunk that declares 80ms while carrying 40ms of samples yields two frames, both
    driven by real audio -- the renderer spreads what it has across the time it was
    told the audio occupies. Trusting the payload instead would drift the mouth out of
    sync with playback, since the transport schedules on duration.
    """
    from avatar.contracts import AudioChunk

    renderer = StubRenderer(width=32, height=32, frame_interval_ms=40)
    session = renderer.start_session(renderer.prepare_identity("ref.mp4"))

    renderer.push_audio(session, AudioChunk(pcm=pcm_at_rms(40, 0.28), epoch=1, duration_ms=80))
    frames = [f.data for f in renderer.frames(session)]

    assert len(frames) == 2
    closed = draw_placeholder(32, 32, mouth_level(b""))
    assert all(f != closed for f in frames), "both frames had audio behind them"


def test_stub_closes_the_mouth_when_its_audio_runs_out() -> None:
    """A frame that is due but has no samples behind it must not hold the mouth open."""
    from avatar.contracts import AudioChunk

    renderer = StubRenderer(width=32, height=32, frame_interval_ms=40)
    session = renderer.start_session(renderer.prepare_identity("ref.mp4"))

    # First chunk establishes the bytes-per-frame ratio and is fully consumed.
    renderer.push_audio(session, AudioChunk(pcm=pcm_at_rms(40, 0.28), epoch=1, duration_ms=40))
    list(renderer.frames(session))

    # Second advances the clock with no samples at all.
    renderer.push_audio(session, AudioChunk(pcm=b"", epoch=1, duration_ms=40))
    starved = [f.data for f in renderer.frames(session)]

    assert starved == [draw_placeholder(32, 32, mouth_level(b""))]


def test_stub_reset_is_safe_with_nothing_in_flight() -> None:
    renderer = StubRenderer()
    session = renderer.start_session(renderer.prepare_identity("ref.mp4"))

    renderer.reset(session)
    renderer.reset(session)

    assert renderer.sessions_closed == 0


def test_stub_rejects_use_after_close() -> None:
    from avatar.contracts import AudioChunk

    renderer = StubRenderer()
    session = renderer.start_session(renderer.prepare_identity("ref.mp4"))
    renderer.close_session(session)

    with pytest.raises(RuntimeError, match="closed"):
        renderer.push_audio(session, AudioChunk(pcm=b"", epoch=1, duration_ms=40))
