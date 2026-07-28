"""
A twenty-line BMP encoder.

Here so that the stub renderer and the placeholder idle loop can both produce frames
a browser will render without pulling Pillow, OpenCV, or numpy into the dependency
set. BMP is the only common format that encodes trivially by hand, and every browser
decodes it.

The real renderer will emit JPEG or WebP from whatever its model already produces --
BMP frames are roughly forty times larger, which is fine on localhost and absurd
over a network. This module does not survive M2 in the hot path.
"""

from __future__ import annotations

import struct

_DIB_HEADER_SIZE = 40
_FILE_HEADER_SIZE = 14


def solid_bmp(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """24-bit BMP, top-down, one flat colour."""
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {width}x{height}")
    r, g, b = rgb
    row = bytes((b, g, r)) * width  # BMP pixel order is BGR
    row += b"\x00" * (-len(row) % 4)  # rows are 4-byte aligned
    pixels = row * height
    # Negative height means the rows are stored top-down, which saves the caller
    # having to reverse them.
    dib = struct.pack(
        "<IiiHHIIiiII",
        _DIB_HEADER_SIZE,
        width,
        -height,
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
