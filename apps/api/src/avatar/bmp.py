"""
A tiny BMP rasteriser: solid fills and rectangles, nothing else.

Here so that the stub renderer and the placeholder idle loop can produce frames a
browser will render without pulling in Pillow, OpenCV, or numpy. BMP is the only
common format that encodes trivially by hand, and every browser decodes it.

Rectangles rather than just solid colours because a solid rectangle is not a
watchable placeholder. The point of the stub is to prove the interface and to make
"is audio driving video?" answerable at a glance, and it cannot do the second job if
every frame looks identical.

Rows are painted by slice assignment, which runs at C speed. Callers are still
expected to cache: at 25fps, rasterising every frame from scratch in Python is
affordable but pointless when the output only takes a dozen distinct forms.

The real renderer will emit JPEG or WebP from whatever its model already produces.
BMP frames are roughly forty times larger, which is fine on localhost and absurd over
a network. This module does not survive M2 in the hot path.
"""

from __future__ import annotations

import struct

_DIB_HEADER_SIZE = 40
_FILE_HEADER_SIZE = 14

RGB = tuple[int, int, int]


class Canvas:
    """A 24-bit RGB raster that knows how to fill rectangles and emit a BMP."""

    def __init__(self, width: int, height: int, background: RGB) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"width and height must be positive, got {width}x{height}")
        self.width = width
        self.height = height
        # BMP rows are 4-byte aligned; the pad bytes sit after the pixel data.
        self._stride = (width * 3 + 3) & ~3
        blank = self._row_bytes(background)
        self._rows = [bytearray(blank) for _ in range(height)]

    def _row_bytes(self, colour: RGB) -> bytes:
        r, g, b = colour
        row = bytes((b, g, r)) * self.width  # BMP pixel order is BGR
        return row + b"\x00" * (self._stride - len(row))

    def fill_rect(self, x: int, y: int, w: int, h: int, colour: RGB) -> None:
        """Fill a rectangle, clipped to the canvas. Silently ignores empty rects."""
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.width, x + w), min(self.height, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        r, g, b = colour
        span = bytes((b, g, r)) * (x1 - x0)
        for row in self._rows[y0:y1]:
            row[x0 * 3 : x1 * 3] = span

    def to_png(self) -> bytes:
        """
        The same image, deflate-compressed and roughly 40x smaller.

        This is the wire format. `to_bmp` is kept because it is what the tests were
        written against and because it is the one encoder with no compression to be
        wrong about -- a useful thing to compare against when a frame looks off.
        """
        from avatar.png import encode

        rows = []
        span = self.width * 3
        for row in self._rows:
            rgb = bytearray(span)
            rgb[0::3] = row[2:span:3]  # BMP stores BGR; PNG wants RGB
            rgb[1::3] = row[1:span:3]
            rgb[2::3] = row[0:span:3]
            rows.append(bytes(rgb))
        return encode(self.width, self.height, rows)

    def to_bmp(self) -> bytes:
        pixels = b"".join(self._rows)
        # Negative height stores rows top-down, so row 0 is the top and callers do
        # not have to reverse anything.
        dib = struct.pack(
            "<IiiHHIIiiII",
            _DIB_HEADER_SIZE,
            self.width,
            -self.height,
            1,
            24,
            0,
            len(pixels),
            2835,
            2835,
            0,
            0,
        )
        offset = _FILE_HEADER_SIZE + len(dib)
        header = b"BM" + struct.pack("<IHHI", offset + len(pixels), 0, 0, offset)
        return header + dib + pixels


def solid_bmp(width: int, height: int, rgb: RGB) -> bytes:
    """24-bit BMP, top-down, one flat colour."""
    return Canvas(width, height, rgb).to_bmp()
