"""
A minimal PNG encoder: 8-bit truecolour, no interlacing, nothing else.

Here because the stub's frames were going over the wire as uncompressed BMP, which
`bmp.py` correctly called "fine on localhost and absurd over a network" -- and then it
was measured: **108 KB per frame, 22.2 Mbps at 25fps.** Through a Cloudflare tunnel to a
Colab runtime, 0.5fps of 25 arrived. The renderer was never the bottleneck; the wire was.

PNG rather than JPEG, for three reasons that all point the same way:

- **No new dependency.** `zlib` is stdlib. Pillow is in `tests/test_boundaries.py`'s
  FORBIDDEN_ROOTS, and while the stub is outside the enforced set today, adding a
  compiled image library to the demo path to save bytes on five rectangles is the
  opposite of boring.
- **It is smaller here.** These frames are flat colour blocks. Lossless PNG beats JPEG
  on flat art both in size and in fidelity, and JPEG would add ringing around the mouth
  edge -- artefacts on the one thing a viewer is being asked to watch.
- **It stays honest.** Lossless means what the browser paints is exactly what the
  renderer produced, so nothing measured here is measuring a compression artefact.

The real renderer's frames are photographic, where JPEG wins by a wide margin. `M2`
should encode with whatever its model pipeline already links against -- Pillow or
OpenCV, both of which it is free to import -- and the client sniffs the format from its
magic bytes, so switching needs no protocol change and no coordinated deploy.

Not implemented, deliberately: palettes, alpha, 16-bit, interlacing, Paeth/Average
filters. A partial implementation of a format is worse than an obviously bounded one,
because the failure is a subtly wrong image rather than an error.
"""

from __future__ import annotations

import struct
import zlib

SIGNATURE = b"\x89PNG\r\n\x1a\n"
"""Magic bytes. The client sniffs these to pick a Blob type, so they are load-bearing."""

_COLOUR_TYPE_RGB = 2
_BIT_DEPTH = 8
_COMPRESSION_LEVEL = 9

FILTER_NONE = 0
FILTER_SUB = 1


def _chunk(tag: bytes, payload: bytes) -> bytes:
    """One PNG chunk: length, tag, payload, CRC over tag+payload."""
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def _filter_sub(row: bytes) -> bytes:
    """
    Each byte minus the one three bytes to its left (the same channel, previous pixel).

    On a run of identical pixels this produces zeros, which deflate encodes almost for
    free. That is the whole reason a 256x144 frame of flat rectangles collapses to a
    couple of KB rather than a couple of hundred.
    """
    out = bytearray(row)
    for i in range(3, len(row)):
        out[i] = (row[i] - row[i - 3]) & 0xFF
    return bytes(out)


def encode(width: int, height: int, rows: list[bytes]) -> bytes:
    """
    Encode top-down RGB scanlines. Each row must be exactly `width * 3` bytes.

    Filtering is adaptive per row: both filters are tried and the smaller result kept.
    That is affordable because callers cache -- the stub rasterises one frame per mouth
    level, not one per frame -- and it is a real win on rows that are a single colour.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {width}x{height}")
    if len(rows) != height:
        raise ValueError(f"expected {height} rows, got {len(rows)}")

    expected = width * 3
    filtered = bytearray()
    for y, row in enumerate(rows):
        if len(row) != expected:
            raise ValueError(f"row {y} is {len(row)} bytes, expected {expected}")
        # Try both, keep the smaller. Ties go to None, which is cheaper to decode.
        sub = _filter_sub(row)
        candidates = (
            (len(zlib.compress(row, 1)), FILTER_NONE, row),
            (len(zlib.compress(sub, 1)), FILTER_SUB, sub),
        )
        _, filter_type, data = min(candidates, key=lambda c: (c[0], c[1]))
        filtered.append(filter_type)
        filtered += data

    ihdr = struct.pack(">IIBBBBB", width, height, _BIT_DEPTH, _COLOUR_TYPE_RGB, 0, 0, 0)
    return (
        SIGNATURE
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(filtered), _COMPRESSION_LEVEL))
        + _chunk(b"IEND", b"")
    )


def decode(data: bytes) -> tuple[int, int, list[bytes]]:
    """
    Decode what `encode` produces, back to RGB scanlines.

    Exists for the tests rather than for the server: it is the only way to assert that a
    PNG frame carries the same pixels as the BMP it replaced without taking a decoder on
    faith or installing one. It handles only the subset `encode` emits.
    """
    if not data.startswith(SIGNATURE):
        raise ValueError("not a PNG")
    pos = len(SIGNATURE)
    width = height = 0
    idat = bytearray()
    while pos < len(data):
        (length,) = struct.unpack_from(">I", data, pos)
        tag = data[pos + 4 : pos + 8]
        payload = data[pos + 8 : pos + 8 + length]
        expected_crc = struct.unpack_from(">I", data, pos + 8 + length)[0]
        if zlib.crc32(tag + payload) & 0xFFFFFFFF != expected_crc:
            raise ValueError(f"CRC mismatch in {tag!r} chunk")
        if tag == b"IHDR":
            width, height, depth, colour, _, _, interlace = struct.unpack(">IIBBBBB", payload)
            if (depth, colour, interlace) != (_BIT_DEPTH, _COLOUR_TYPE_RGB, 0):
                raise ValueError("only 8-bit non-interlaced truecolour is supported")
        elif tag == b"IDAT":
            idat += payload
        elif tag == b"IEND":
            break
        pos += 12 + length

    raw = zlib.decompress(bytes(idat))
    stride = width * 3
    rows: list[bytes] = []
    pos = 0
    for _ in range(height):
        filter_type = raw[pos]
        row = bytearray(raw[pos + 1 : pos + 1 + stride])
        if filter_type == FILTER_SUB:
            for i in range(3, stride):
                row[i] = (row[i] + row[i - 3]) & 0xFF
        elif filter_type != FILTER_NONE:
            raise ValueError(f"unsupported filter {filter_type}")
        rows.append(bytes(row))
        pos += 1 + stride
    return width, height, rows
