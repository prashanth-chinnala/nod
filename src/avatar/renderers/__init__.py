"""
Renderer registry.

`build` is the one-line swap the model-selection memo points at: changing which
ML model drives the avatar is a change to a `RendererConfig` value and nothing
else. No orchestration module imports this package, so a renderer that drags in
torch cannot pull it into the state machine's import graph.

Registration is lazy on purpose. Importing a GPU renderer costs seconds and needs
weights on disk; doing it at package-import time would make `import avatar` fail
on a laptop and drag CUDA into CI.
"""

from __future__ import annotations

from avatar.contracts import RendererConfig, TalkingHeadRenderer


def build(config: RendererConfig | None = None) -> TalkingHeadRenderer:
    config = config or RendererConfig()
    name = config.name.lower()

    if name == "stub":
        from avatar.renderers.stub import StubRenderer

        return StubRenderer(**config.options)  # type: ignore[arg-type]

    # M2 lands the chosen model here, e.g.:
    #   if name == "musetalk":
    #       from avatar.renderers.musetalk import MuseTalkRenderer
    #       return MuseTalkRenderer(**config.options)

    raise ValueError(
        f"unknown renderer {config.name!r}; available: 'stub'. "
        "The real renderer is pending the M0 model spike."
    )


__all__ = ["build"]
