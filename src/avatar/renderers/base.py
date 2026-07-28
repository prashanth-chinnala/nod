"""
Re-export of the renderer contract.

Implementations import `TalkingHeadRenderer` from here rather than from
`avatar.contracts`, so that a renderer module's import list names only the
renderer package. It is a one-line indirection that makes the dependency
direction obvious in review: renderers depend on the contract, never the reverse,
and nothing in the orchestration layer imports this package at all.
"""

from __future__ import annotations

from avatar.contracts import AudioChunk, Frame, TalkingHeadRenderer

__all__ = ["AudioChunk", "Frame", "TalkingHeadRenderer"]
