"""
Every renderer must accept what the server passes it.


**Why this file exists.** `tests/test_musetalk_renderer.py` asserts `isinstance(renderer,
TalkingHeadRenderer)`, and `MuseTalkRenderer`'s own docstring claimed that conformance was
checked "so this cannot subclass its way to compliance while quietly taking different
arguments". It could, and it did. A `runtime_checkable` Protocol compares *method names* -- not
their signatures, and not the constructor at all -- so `AVATAR_RENDERER=musetalk` raised


TypeError: MuseTalkRenderer.__init__() got an unexpected keyword argument 'width'


with a full green suite behind it. The failure landed at the worst possible moment, in
`BrowserSession.run`, as a rejected WebSocket at the instant a candidate opened their link.


**What this guarantees, and it is the point of keeping the stub safe.** The stub is the renderer
CI runs, the one every one of the other tests uses, and the fallback when a GPU is missing or
the weights are absent. Work on the real renderer must not be able to break it, and the two must
stay interchangeable at the seam `build()` defines. So both are constructed here from the
*server's own* option dict, imported rather than copied -- a test with its own hardcoded copy of
those keys would pass happily while the server passed something else.


Adding a renderer means adding it to `NAMES` and making it accept `renderer_options()`. A
renderer may ignore any option it has no use for; it may not refuse one.
"""

from __future__ import annotations

import inspect

import pytest

from avatar.contracts import RendererConfig, TalkingHeadRenderer
from avatar.renderers import build
from avatar.server import renderer_options

NAMES = ("stub", "musetalk")
"""
Every renderer `build()` accepts.


`musetalk` is safe to construct in CI with no torch and no weights: the backend and its several
GB are created on first use, not in `__init__`. That laziness is a load-bearing property of the
module boundary, and constructing it here is also the only test that keeps it honest.
"""


@pytest.mark.parametrize("name", NAMES)
def test_every_renderer_accepts_the_servers_options(name: str) -> None:
    """The regression. Each renderer builds from exactly what the server passes."""
    renderer = build(RendererConfig(name=name, options=renderer_options()))
    assert isinstance(renderer, TalkingHeadRenderer)


@pytest.mark.parametrize("name", NAMES)
def test_every_renderer_accepts_each_option_alone(name: str) -> None:
    """
    One option at a time, so a failure names the option rather than the dict.

    `TypeError: unexpected keyword argument 'width'` is a clear message; it is much less clear
    when four keys are passed together and the report says only that construction failed.
    """
    for key, value in renderer_options().items():
        build(RendererConfig(name=name, options={key: value}))


@pytest.mark.parametrize("name", NAMES)
def test_every_renderer_builds_with_no_options_at_all(name: str) -> None:
    """
    Defaults for everything, which is how `POST /faces/{id}/prepare` constructs one.

    That endpoint passes no options, so a renderer with a required keyword argument would enroll
    nothing while the session path kept working -- a split failure, harder to place than a total
    one.
    """
    assert isinstance(build(RendererConfig(name=name)), TalkingHeadRenderer)


@pytest.mark.parametrize("name", NAMES)
def test_no_renderer_takes_a_required_constructor_argument(name: str) -> None:
    """
    Every parameter has a default, and every one is keyword-only.


    Checked against the signature rather than by construction, because this is the property
    that makes the two renderers substitutable at all: `build()` passes a dict of keyword
    arguments and nothing else, so a positional-or-required parameter is unreachable through
    the seam. Catching it here names the parameter, where catching it by construction only
    says `TypeError`.
    """
    renderer_type = type(build(RendererConfig(name=name)))
    for parameter in inspect.signature(renderer_type.__init__).parameters.values():
        if parameter.name == "self":
            continue
        assert parameter.default is not inspect.Parameter.empty, (
            f"{renderer_type.__name__}.{parameter.name} has no default, so `build()` cannot "
            "construct this renderer without knowing about it specifically"
        )
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{renderer_type.__name__}.{parameter.name} is not keyword-only; `build()` only "
            "ever passes keyword arguments"
        )


def test_an_unknown_renderer_is_refused_by_name() -> None:
    """
    A typo in `AVATAR_RENDERER` must say so, and say what the choices are.


    Silently falling back to the stub would be the harmful alternative: a deployment that
    believes it is running the real renderer and is not, which is precisely the confusion
    `enrollment_ms: 0` caused when enrollment was pinned to the stub.
    """
    with pytest.raises(ValueError, match="unknown renderer") as caught:
        build(RendererConfig(name="wav2lip"))
    for name in NAMES:
        assert name in str(caught.value)
