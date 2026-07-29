#!/usr/bin/env python3
"""
Generate `apps/web/src/app/favicon.ico` from the identity's app-icon geometry.

**Why a script and not a checked-in binary alone.** The .ico *is* checked in, because a build must
not depend on running this. But a binary asset nobody can regenerate is a mystery: the next person
who needs to shift the lamp colour has to open a graphics editor and guess at the geometry. This
file is the source, the .ico is the artefact, and the numbers below are the identity's.

**Why not `avatar.png`.** The project's PNG encoder states plainly that alpha is *not* implemented,
deliberately, and it is right to: it exists to put video frames on a wire, where every frame is
opaque. A favicon needs alpha, because the tile has rounded corners and dark square corners against
a light browser tab is exactly the tell of an icon nobody looked at. Widening that module for a
one-off asset would trade a clean boundary for nothing. So the RGBA writer lives here, where the
one-off thing is.

**Why two drawings rather than one scaled.** The identity specifies a different 16px mark — one
circle instead of two — because at that size the lamp and its ghost merge into a single grey smudge
and the nod is lost. A browser chooses which size it wants and cannot be asked to prefer the good
one, so both are embedded and each is drawn for the size it will be seen at.

    python scripts/make_favicon.py
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "apps/web/src/app/favicon.ico"

TILE = (0x13, 0x1B, 0x21)
LAMP = (0x54, 0xB6, 0xD6)
GHOST_ALPHA = 0.22
"""The identity's tile, lamp, and ghost opacity. Literal here because an .ico has no stylesheet."""

SS = 4
"""
Supersampling factor.

Antialiasing by averaging a 4x4 grid per pixel rather than computing coverage analytically. At
16x16 the difference between a smooth circle and a jagged one is the whole impression, and 16 sub
samples is enough that no stair-stepping survives while staying instant to compute.
"""


def _rounded_rect_coverage(x: float, y: float, size: float, radius: float) -> bool:
    """Whether a point falls inside a rounded square of `size` with corner `radius`."""
    if radius <= 0:
        return True
    # Only the four corner boxes need a distance test; everything else is inside by inspection.
    for cx, cy in (
        (radius, radius),
        (size - radius, radius),
        (radius, size - radius),
        (size - radius, size - radius),
    ):
        in_x = x < radius if cx == radius else x > size - radius
        in_y = y < radius if cy == radius else y > size - radius
        if in_x and in_y:
            return (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
    return True


def _blend(under: tuple[int, int, int], over: tuple[int, int, int], alpha: float) -> tuple[int, ...]:
    return tuple(round(u + (o - u) * alpha) for u, o in zip(under, over, strict=True))


def render(size: int, circles: list[tuple[float, float, float, float]], radius: float) -> bytes:
    """
    Rasterise one icon to RGBA rows.

    `circles` is (cx, cy, r, alpha) in *icon* pixels, painted in order, so a ghost listed before
    the lamp is overlapped by it rather than the reverse.
    """
    rows = bytearray()
    for py in range(size):
        for px in range(size):
            inside = 0
            colour_acc = [0.0, 0.0, 0.0]
            for sy in range(SS):
                for sx in range(SS):
                    x = px + (sx + 0.5) / SS
                    y = py + (sy + 0.5) / SS
                    if not _rounded_rect_coverage(x, y, size, radius):
                        continue
                    inside += 1
                    pixel: tuple[int, ...] = TILE
                    for cx, cy, r, alpha in circles:
                        if (x - cx) ** 2 + (y - cy) ** 2 <= r**2:
                            pixel = _blend(tuple(pixel), LAMP, alpha)  # type: ignore[arg-type]
                    for channel in range(3):
                        colour_acc[channel] += pixel[channel]
            total = SS * SS
            if inside == 0:
                rows += bytes((0, 0, 0, 0))
                continue
            # Colour is the mean over *covered* samples only. Averaging over all of them would
            # darken the rounded corners toward black, which is the classic haloed-icon artefact.
            rows += bytes(
                (
                    round(colour_acc[0] / inside),
                    round(colour_acc[1] / inside),
                    round(colour_acc[2] / inside),
                    round(255 * inside / total),
                )
            )
    return bytes(rows)


def png_rgba(size: int, pixels: bytes) -> bytes:
    """Minimal 8-bit RGBA PNG. No filtering: at these sizes it saves nothing worth the code."""
    raw = bytearray()
    stride = size * 4
    for y in range(size):
        raw += b"\x00"  # filter type 0, none
        raw += pixels[y * stride : (y + 1) * stride]

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)  # colour type 6 = RGBA
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def ico(images: list[tuple[int, bytes]]) -> bytes:
    """
    Wrap PNGs in an ICO container.

    PNG payloads rather than the older DIB form: every browser in use has supported them for well
    over a decade, and a DIB would need its own bottom-up rows and an AND mask to hand-build for no
    gain.
    """
    count = len(images)
    header = struct.pack("<HHH", 0, 1, count)
    offset = len(header) + 16 * count
    directory = bytearray()
    body = bytearray()
    for size, data in images:
        directory += struct.pack(
            "<BBBBHHII",
            size if size < 256 else 0,  # 0 means 256 in this field
            size if size < 256 else 0,
            0,  # palette size, 0 for truecolour
            0,  # reserved
            1,  # colour planes
            32,  # bits per pixel
            len(data),
            offset,
        )
        body += data
        offset += len(data)
    return header + bytes(directory) + bytes(body)


def main() -> None:
    # 32px: the full mark. Geometry is the identity's 64px drawing halved.
    large = render(
        32,
        [(16, 11, 4, GHOST_ALPHA), (16, 20.5, 4, 1.0)],
        radius=15 / 2,
    )
    # 16px: one circle, per the identity's own small cut. Its 64px source is cx32 cy34 r13.
    small = render(16, [(8, 8.5, 3.25, 1.0)], radius=15 / 4)

    payload = ico([(16, png_rgba(16, small)), (32, png_rgba(32, large))])
    OUT.write_bytes(payload)
    print(f"wrote {OUT.relative_to(Path.cwd())} ({len(payload):,} bytes, 16px + 32px)")


if __name__ == "__main__":
    main()
