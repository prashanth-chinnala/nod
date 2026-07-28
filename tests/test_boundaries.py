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
import sys
from pathlib import Path

import pytest

from avatar import contracts, mixer, orchestrator, state, telemetry
from avatar.contracts import RendererConfig, TalkingHeadRenderer
from avatar.renderers import build
from avatar.renderers.stub import StubRenderer

ORCHESTRATION_MODULES = (contracts, state, telemetry, mixer, orchestrator)

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
    """Belt to the AST braces: nothing arrives transitively either."""
    loaded = set(sys.modules)
    offenders = loaded & FORBIDDEN_ROOTS

    assert not offenders, f"importing avatar loaded {sorted(offenders)}"


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
    assert all(f.data.startswith(b"BM") for f in frames), "valid BMP payloads"


def test_stub_withholds_frames_until_its_lookahead_window_is_filled() -> None:
    from avatar.contracts import AudioChunk

    renderer = StubRenderer(width=4, height=4, first_frame_delay_ms=200, frame_interval_ms=40)
    session = renderer.start_session(renderer.prepare_identity("ref.mp4"))

    renderer.push_audio(session, AudioChunk(pcm=b"", epoch=1, duration_ms=160))
    assert list(renderer.frames(session)) == []

    renderer.push_audio(session, AudioChunk(pcm=b"", epoch=1, duration_ms=80))
    assert len(list(renderer.frames(session))) == 1


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
