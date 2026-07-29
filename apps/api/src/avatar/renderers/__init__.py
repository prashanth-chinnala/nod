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

    if name == "musetalk":
        # The import is inside the branch, so selecting 'stub' never touches this module
        # and CI never needs torch. Constructing the renderer is still cheap -- the backend
        # and its several GB of weights load on first use, not here.
        from avatar.renderers.musetalk import MuseTalkRenderer

        return MuseTalkRenderer(**config.options)  # type: ignore[arg-type]

    raise ValueError(
        f"unknown renderer {config.name!r}; available: 'stub', 'musetalk'. "
        "'musetalk' additionally needs the model weights on disk and a GPU."
    )


__all__ = ["build"]
