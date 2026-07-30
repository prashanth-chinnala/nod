"""
Uploaded reference media: where it lives, whether it is usable, and what it looks like.

**Why the upload is a permanent input, not a temporary one.** With a reference-driven renderer
the final video *is* the reference video with the mouth repainted — `musetalk_torch.prepare()`
keeps `frames` in the identity artifact, and every rendered frame is one of them with a new
lower face blended in. So this file is not consumed at preparation time and discarded; it is the
visual source of every frame of every session that uses the face, for as long as the face
exists. That is the difference between this and a trained-replica product, where the upload is
training data and nothing of it survives into the output.

Which decides the storage: files on disk, not bytes in a column. A 20 MB clip in a JSONB
document would be read and rewritten on every unrelated patch to the same record, and the same
reasoning keeps recordings out of the database.

**Why ffmpeg rather than a Python decoder.** It is already a hard requirement of the renderer —
MuseTalk takes an `--ffmpeg_path` argument — so it is not a new dependency, it is one that was
already implied. Probing with `ffprobe` also means the answer comes from the thing that will
actually read the file later, rather than from a second decoder that might disagree.

**Why a still image becomes a clip.** A photograph has no head motion, and a reference-driven
renderer has nothing to borrow: the result is a rigid head with a moving mouth. Rather than let
that difference leak into every downstream stage, an image is expanded here into a short clip so
`prepare_identity` sees one uniform kind of input. What the expansion cannot do is invent
motion, and the console says so — an animated portrait needs a model that generates movement,
which is a different renderer, not a different file format.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

MEDIA_ROOT = Path(os.environ.get("AVATAR_MEDIA_DIR", "media"))
"""
Where uploaded references and their thumbnails live.

Relative by default, like `AVATAR_DATA_DIR`, and carrying the same trap: it resolves against the
process's working directory, so two services started from different places would disagree about
where a face's video is. `.env.development` sets it absolutely for that reason. This has already
cost three separate debugging sessions in this repo -- a store at the repo root, an assistant
reading an empty directory, and an API serving JSON while claiming Postgres -- so it is stated
here rather than left to be rediscovered a fourth time.
"""

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
"""
200 MB. Generous for a reference clip and far below what a stuck upload would send.

A reference is tens of seconds of one person's head. Anything approaching this is the wrong
file, and a bound is what stops the wrong file from filling a disk before anyone notices.
"""

MIN_VIDEO_SECONDS = 5.0
RECOMMENDED_VIDEO_SECONDS = 20.0
"""
The reference loops, so its length is visible in the product rather than an implementation
detail.

`prepare()` cycles the frames forward then backward, so a clip of length N repeats every 2N
seconds. At 5 seconds that is a 10-second loop and a candidate will notice; below 5 there is not
enough motion to be worth borrowing at all. 20 is the point where the repeat stops drawing
attention. Refused below the minimum, warned below the recommendation -- an operator who only
has 8 seconds should get a face, plus the reason it will look repetitive.
"""

IMAGE_CLIP_SECONDS = 4.0
"""
How long a clip generated from a still image runs.

Short on purpose. It has no motion, so a longer clip buys nothing but disk -- the loop is
imperceptible either way because every frame is identical.
"""

VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".m4v"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class MediaRejected(ValueError):
    """
    The upload cannot be used as a reference, with a reason an operator can act on.

    Its own type so a router answers 422 with the reason rather than 500 with a traceback: every
    case here is something about the file the person who chose it can fix.
    """


@dataclass(frozen=True)
class Probe:
    """What ffprobe found. `duration` is None for a still image."""

    kind: str
    duration: float | None
    width: int
    height: int


def _ffmpeg(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise MediaRejected(
            f"{name} is not installed, and reference media cannot be validated without it. "
            "`brew install ffmpeg` (macOS) or the equivalent. The renderer needs it too."
        )
    return path


def probe(path: Path) -> Probe:
    """
    Ask ffprobe what this file is.

    Read from the file rather than trusted from its extension or the browser's content type,
    both of which are claims made by the uploader. A `.mp4` containing a PDF is the case that
    matters: it passes every check based on the name and fails inside the renderer, hours later,
    as a preparation error nobody can trace back to the upload.
    """
    result = subprocess.run(
        [
            _ffmpeg("ffprobe"),
            "-v", "error",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise MediaRejected(
            "ffprobe could not read this file, so it is not usable video or image: "
            + (result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "no detail")
        )

    info = json.loads(result.stdout or "{}")
    video = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "video"), None
    )
    if video is None:
        raise MediaRejected("this file has no video or image stream in it")

    # Image or video, from three signals because no single one covers every case. A PNG reports
    # NO `nb_frames`, NO stream duration and NO format duration -- so a frame-count check alone
    # classified a still as a video of zero length and rejected it as "too short", which
    # is a confusing way to refuse a perfectly good photograph.
    #
    # `format_name` is the reliable part: ffmpeg reads stills through demuxers named `png_pipe`,
    # `jpeg_pipe`, `webp_pipe` and so on, while a real container reports something like
    # `mov,mp4,m4a`. The other two checks stay because they catch a single-frame mp4, which is a
    # legal video file that should still be treated as a still.
    fmt = info.get("format") or {}
    frames = video.get("nb_frames")
    duration_raw = fmt.get("duration")
    duration = float(duration_raw) if duration_raw not in (None, "N/A") else None
    single_frame = (
        str(fmt.get("format_name", "")).endswith("_pipe")
        or frames in ("1", 1)
        or duration is None
        or duration < 0.1
    )

    return Probe(
        kind="image" if single_frame else "video",
        duration=None if single_frame else duration,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
    )


def check(found: Probe) -> list[str]:
    """
    Refuse what cannot work; warn about what will work badly. Returns the warnings.

    The split matters. A refusal is for a file the renderer cannot use at all; a warning is for
    one that produces a face someone will be unhappy with. Turning the second into the first
    would tell an operator with an 8-second clip that they cannot have a face, which is not true
    -- they can have a repetitive one, and they should be the one deciding whether that is
    acceptable.
    """
    if found.width < 256 or found.height < 256:
        raise MediaRejected(
            f"the face is {found.width}x{found.height}; the renderer crops and encodes a face "
            "region at 256x256, so anything smaller is upscaled and looks it. Use at least "
            "512 on the short side."
        )

    warnings: list[str] = []
    if found.kind == "video":
        seconds = found.duration or 0.0
        if seconds < MIN_VIDEO_SECONDS:
            raise MediaRejected(
                f"the clip is {seconds:.1f}s. The reference loops -- forward then backward, "
                f"so it repeats every {2 * seconds:.0f}s -- and below "
                f"{MIN_VIDEO_SECONDS:.0f}s there is not enough motion to borrow. Record at "
                f"least {RECOMMENDED_VIDEO_SECONDS:.0f}s of the person sitting still and "
                "looking ahead."
            )
        if seconds < RECOMMENDED_VIDEO_SECONDS:
            warnings.append(
                f"{seconds:.0f}s of reference loops every {2 * seconds:.0f}s, which a "
                f"candidate will notice. {RECOMMENDED_VIDEO_SECONDS:.0f}s or more stops it "
                "drawing attention."
            )
    else:
        warnings.append(
            "a still image has no head motion for the renderer to borrow, so this persona "
            "will hold one pose with a moving mouth. A short video of the same person looks "
            "markedly more alive."
        )
    if found.width < 512 or found.height < 512:
        warnings.append(
            f"{found.width}x{found.height} is usable but tight; the face crop is 256x256, so a "
            "small source leaves little detail in the part that moves."
        )
    return warnings


def store_upload(data: bytes, filename: str) -> Path:
    """
    Write an upload under `MEDIA_ROOT` with a generated name, and return the path.

    The uploader's filename is used only for its suffix. Anything else from it -- directories,
    `..`, a leading dot, a 300-character name -- is a path the uploader chose on our disk, and
    the store already refuses ids containing separators for the same reason.
    """
    suffix = Path(filename or "").suffix.lower()
    if suffix not in VIDEO_SUFFIXES | IMAGE_SUFFIXES:
        raise MediaRejected(
            f"{suffix or 'that file'} is not a supported reference. Video: "
            f"{', '.join(sorted(VIDEO_SUFFIXES))}. Image: {', '.join(sorted(IMAGE_SUFFIXES))}."
        )
    if len(data) > MAX_UPLOAD_BYTES:
        raise MediaRejected(
            f"the file is {len(data) / 1_048_576:.0f} MB, over the "
            f"{MAX_UPLOAD_BYTES // 1_048_576} MB ceiling. A reference is tens of seconds of "
            "one person's head; anything this large is the wrong file."
        )
    if not data:
        raise MediaRejected("the upload was empty")

    MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
    target = MEDIA_ROOT / f"ref-{uuid.uuid4().hex[:12]}{suffix}"
    target.write_bytes(data)
    return target


def clip_from_image(image: Path) -> Path:
    """
    Expand a still into a short clip, so everything downstream sees one kind of input.

    Scaled to an even width and height because H.264 requires it and an odd dimension fails at
    the encoder with a message about chroma subsampling that reads as a corrupt file.

    This does not animate anything. It exists so `prepare_identity`, the frame cycling, and the
    render window arithmetic have no special case for images -- not to make a photograph move.
    """
    target = image.with_name(image.stem + "-still.mp4")
    result = subprocess.run(
        [
            _ffmpeg("ffmpeg"), "-y",
            "-loop", "1", "-i", str(image),
            "-t", str(IMAGE_CLIP_SECONDS),
            "-r", "25",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            str(target),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        raise MediaRejected(
            "could not build a clip from this image: "
            + (result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "no detail")
        )
    return target


def thumbnail(reference: Path) -> Path:
    """
    One frame, for the console to show. Taken a second in, not at zero.

    The first frame of a phone recording is frequently the moment before the person settled --
    mid-blink, looking away, or still black from the sensor warming up. A second in is past that
    and still cheap. Falls back to the first frame for a clip shorter than that.
    """
    target = reference.with_name(reference.stem + "-thumb.jpg")
    for seek in ("1", "0"):
        result = subprocess.run(
            [
                _ffmpeg("ffmpeg"), "-y",
                "-ss", seek, "-i", str(reference),
                "-frames:v", "1",
                "-vf", "scale=480:-2",
                str(target),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0 and target.exists() and target.stat().st_size > 0:
            return target
    raise MediaRejected("could not extract a preview frame from this file")
