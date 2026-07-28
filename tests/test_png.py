"""
The PNG encoder, and the size claim that justifies it existing.

The reason this module exists is a measurement, not a preference: BMP frames were 108 KB
each, 22.2 Mbps at 25fps, and through a tunnel to a Colab runtime 0.5fps of 25 arrived.
So the tests here are in two halves -- that the bytes decode to exactly the pixels the
BMP carried (a lossy or wrong encoder would be far worse than a large one), and that the
size actually collapsed. The second half is a real assertion rather than a comment,
because "we added compression" is worth nothing without a ratio.
"""

from __future__ import annotations

import zlib

import pytest

from avatar.bmp import Canvas
from avatar.png import SIGNATURE, decode, encode
from avatar.renderers.stub import MOUTH_LEVELS, draw_placeholder

GROUND = (18, 27, 34)


def bmp_pixels(canvas: Canvas) -> list[bytes]:
    """The canvas's own rows as top-down RGB, for comparison against a decode."""
    span = canvas.width * 3
    rows = []
    for row in canvas._rows:
        bgr = row[:span]
        rows.append(
            bytes(b for i in range(0, span, 3) for b in (bgr[i + 2], bgr[i + 1], bgr[i]))
        )
    return rows


# -- correctness ------------------------------------------------------------


def test_a_png_frame_carries_exactly_the_pixels_the_bmp_did() -> None:
    """
    Lossless, and byte-identical. The whole argument for PNG over JPEG here.

    If this drifts, every downstream measurement is measuring a compression artefact
    rather than what the renderer produced.
    """
    canvas = Canvas(64, 32, GROUND)
    canvas.fill_rect(8, 4, 20, 10, (200, 100, 50))
    canvas.fill_rect(0, 0, 3, 3, (255, 255, 255))
    canvas.fill_rect(60, 28, 10, 10, (1, 2, 3))  # deliberately clipped

    width, height, rows = decode(canvas.to_png())

    assert (width, height) == (64, 32)
    assert rows == bmp_pixels(canvas)


def test_it_starts_with_the_png_magic_bytes() -> None:
    """The client sniffs these to choose a Blob type, so they are load-bearing."""
    assert draw_placeholder(32, 32, 0).startswith(SIGNATURE)
    assert draw_placeholder(32, 32, 0, fmt="bmp").startswith(b"BM")


@pytest.mark.parametrize("size", [(1, 1), (4, 4), (17, 3), (256, 144)])
def test_odd_sizes_round_trip(size: tuple[int, int]) -> None:
    """
    Widths that are not multiples of four are the interesting case.

    BMP pads rows to a 4-byte boundary and PNG does not, so a converter that forgot to
    strip the padding would produce a skewed image -- and it would only skew at some
    widths, which is the kind of bug that survives a single test.
    """
    width, height = size
    canvas = Canvas(width, height, GROUND)
    canvas.fill_rect(0, 0, max(1, width // 2), max(1, height // 2), (250, 10, 10))

    decoded_w, decoded_h, rows = decode(canvas.to_png())

    assert (decoded_w, decoded_h) == (width, height)
    assert rows == bmp_pixels(canvas)


def test_a_corrupted_chunk_is_rejected_rather_than_half_decoded() -> None:
    data = bytearray(draw_placeholder(16, 16, 3))
    data[-6] ^= 0xFF  # inside IEND's CRC region
    with pytest.raises(ValueError, match="CRC"):
        decode(bytes(data))


def test_encode_rejects_rows_that_do_not_match_the_header() -> None:
    """A silently truncated row would shear the image rather than fail."""
    with pytest.raises(ValueError, match="expected 2 rows"):
        encode(2, 2, [b"\x00" * 6])
    with pytest.raises(ValueError, match="row 1 is"):
        encode(2, 2, [b"\x00" * 6, b"\x00" * 3])


def test_the_declared_format_is_validated() -> None:
    """A typo should not silently fall back to the 40x larger encoder."""
    with pytest.raises(ValueError, match="unknown frame format"):
        draw_placeholder(8, 8, 0, fmt="jpg")


# -- the size claim ---------------------------------------------------------


def test_a_demo_frame_is_at_least_ten_times_smaller_than_bmp() -> None:
    """
    The measurement that justifies the module. 22.2 Mbps is what broke the tunnel.

    Ten is a deliberately loose floor -- the real ratio on these frames is far higher --
    because the point is to catch a regression that quietly restores an uncompressed
    wire format, not to pin an exact compressor output across zlib versions.
    """
    png = draw_placeholder(256, 144, 6)
    bmp = draw_placeholder(256, 144, 6, fmt="bmp")

    ratio = len(bmp) / len(png)
    assert ratio >= 10, f"only {ratio:.1f}x smaller: {len(bmp)} -> {len(png)} bytes"


def test_every_mouth_level_stays_small() -> None:
    """
    An open mouth has more edges than a closed one, so it compresses worst.

    Checking the whole range rather than one frame, because a bitrate budget has to hold
    during speech -- which is precisely when the mouth is open and the frames are least
    compressible.
    """
    sizes = [len(draw_placeholder(256, 144, level)) for level in range(MOUTH_LEVELS)]
    worst = max(sizes)
    mbps = worst * 25 * 8 / 1e6
    assert mbps < 3.0, f"worst frame {worst} bytes = {mbps:.1f} Mbps at 25fps"


def test_a_flat_frame_compresses_far_better_than_a_detailed_one() -> None:
    """
    Sanity check on the filter choice, not a requirement.

    If a flat frame did *not* compress better, adaptive filtering would be broken and
    the ratio above would be coming from somewhere unintended.
    """
    flat = len(Canvas(128, 128, GROUND).to_png())
    noisy_canvas = Canvas(128, 128, GROUND)
    for i in range(0, 128, 2):
        noisy_canvas.fill_rect(i, 0, 1, 128, (i * 2 % 256, 255 - i, i))
    noisy = len(noisy_canvas.to_png())

    assert flat < noisy


def test_the_encoder_output_is_deterministic() -> None:
    """Frames are cached by mouth level, so the same level must encode identically."""
    assert draw_placeholder(64, 64, 5) == draw_placeholder(64, 64, 5)


def test_idat_actually_decompresses() -> None:
    """
    Guards the chunk framing independently of `decode`.

    `decode` is our own reader, so a matched pair of bugs in encode and decode could
    agree with each other and pass every test above. Going straight to zlib does not.
    """
    data = draw_placeholder(32, 16, 4)
    start = data.index(b"IDAT") + 4
    length = int.from_bytes(data[start - 8 : start - 4], "big")
    raw = zlib.decompress(data[start : start + length])
    # One filter byte plus width*3 per row.
    assert len(raw) == 16 * (1 + 32 * 3)
