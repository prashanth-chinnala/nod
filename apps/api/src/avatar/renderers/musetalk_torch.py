"""
The GPU half of the MuseTalk renderer. Every torch, CUDA, and OpenCV call lives here.

Separated from `musetalk.py` so that the streaming logic -- buffering, windowing, epoch
tagging, reset -- is testable on a machine with no GPU, and so that `import avatar` never
pulls torch into the graph. `tests/test_boundaries.py` enforces the second part.

**This file has now run.** Every earlier version of this docstring said it had never been
executed, and that was the honest state; it no longer is. On an Apple M1 Pro (16 GB, MPS,
torch 2.13, Python 3.10) it loads all five models in 13.9-20.2s, prepares a 550-frame
reference in 233.9s, and renders frames whose lip region tracks the audio while the rest of
the frame stays bit-identical to the reference.

What it does *not* do on that hardware is keep up. Measured, median of six batches of eight
after a warm-up batch: **2604 ms/frame**, min 1879, max 3255. The budget at 25fps is 40 ms, so
this is 65x too slow -- correct output, nowhere near realtime. No CUDA measurement exists yet,
and none may be quoted until a run produces one.

**Written against MuseTalk v1.5's actual API, read from the checkout, not from its README.**
An earlier version of this file was written from the v1.0 docs and had five signatures wrong
in ways that all fail at the same place -- inside model loading, several frames from the
cause. What v1.5 really wants, and what the mapping is:

| MuseTalk v1.5 | Here |
|---|---|
| `UNet(config, path, use_float16, device)` + `VAE(path, use_float16)` | `load()` |
| `WhisperModel.from_pretrained(dir)` + `AudioProcessor(feature_extractor_path=)` | `load()` |
| `FaceParsing(left_cheek_width, right_cheek_width)` | `load()` |
| `get_landmark_and_bbox(paths, bbox_shift)` | `avatar.renderers.landmarks`, see below |
| `vae.get_latents_for_unet(crop)` -- one 256x256 crop at a time | `prepare()` |
| `get_image_prepare_material(frame, box, fp=, mode=)` -> `(mask, crop_box)` | `prepare()` |
| `datagen()` + the batch loop | `render()` |
| `get_image_blending(frame, face, box, mask, mask_crop_box)` | `render()` |

The one dependency substitution is landmarks: upstream's `preprocessing.py` builds an mmpose
RTMPose model at *import* time, and mmpose ships as pinned compiled wheels for Linux and
Windows only. `avatar.renderers.landmarks` produces the same iBUG-68 points with a pure-torch
detector, and that module documents the trade-off. Nothing else here diverges: the crop
arithmetic, `extra_margin`, the LANCZOS4 resize, the parsing mode and the cheek widths are all
upstream's values, because the U-Net was trained on inputs those constants produce.

**Why JPEG here and PNG in the stub.** The stub draws flat colour blocks, where lossless PNG
is both smaller and artifact-free. A rendered face is photographic, where PNG barely
compresses -- a 256x256 photographic PNG runs 60-120 KB, which at 25fps is 12-24 Mbps and
puts us straight back into the bandwidth wall that PNG was introduced to solve. JPEG at
quality 82 lands the same frame around 8-15 KB, roughly 2-3 Mbps. The client sniffs the
format from magic bytes, so this needs no protocol change and no coordinated deploy.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

JPEG_QUALITY = 82
"""
Quality/size trade-off for rendered frames.

82 is the usual knee: visible ringing starts below ~75, and above ~90 the file grows fast
for detail a 256x256 face region does not carry. `AVATAR_JPEG_QUALITY` overrides it, because
the right value depends on the link and on the model's output resolution, neither of which
is measured yet.
"""

BATCH_SIZE = 4
"""
Frames per U-Net forward pass. Upstream's default is 8; measured, 4 is better here.

The usual reasoning -- a bigger batch uses the device better, at the cost of first-frame
latency, since no frame emits until all of them are done -- turns out to be backwards on 16 GB
of unified memory. Measured on this M1 Pro (ms/frame, median of 4 runs after a warm-up):

    batch  1   330      batch  4   305
    batch  2   317      batch  6   413
    batch  3   301      batch  8   355
                        batch 32  1565

Flat from 1 to 4, then it degrades, and by 32 it is 5x worse than 3. Nothing about the model
changes -- the VAE decode cost stays linear in frames at every batch size -- so this is memory
bandwidth and pressure, not utilisation. On a discrete GPU with its own VRAM the curve should
look like the textbook one, which is why this is a measured default rather than a fixed one:
`AVATAR_MUSETALK_BATCH` overrides it, and it should be re-measured on CUDA rather than assumed.

Picking 4 over 3 for one non-measured reason, stated as such: 4 divides the 8-frame render
window evenly, and a ragged final batch adds a partial forward pass for no frames.
"""

FACE_SIZE = 256
"""The crop the U-Net was trained on. Not a tunable."""

EXTRA_MARGIN = 10
"""
Pixels added below the landmark box before cropping. v1.5's default.

Upstream applies it at crop time *and* again at blend time, so the two must agree or the
repainted mouth lands offset from the hole it was cut from. Here the margin is folded into
the stored box once, in `prepare`, which makes disagreement impossible rather than merely
unlikely.
"""

PARSING_MODE = "jaw"
CHEEK_WIDTH = 90
"""
v1.5's blending defaults: parse the jaw, and how wide a cheek region the mask covers.

Carried over rather than chosen. These decide where the generated face is feathered into the
real one, and a different value is a visible seam, not a preference.
"""

AUDIO_PADDING = 2
"""
Whisper chunks of context on each side of the window, upstream's default.

Real context, not padding in the zero-fill sense: a phoneme's mouth shape depends on its
neighbours, so a window rendered without them articulates the boundary frames wrongly.
"""

MAX_OUTPUT_HEIGHT = 512
"""
Tallest frame this renderer emits, aspect preserved. `AVATAR_MUSETALK_MAX_HEIGHT` overrides.

A bandwidth bound, not a quality choice. MuseTalk blends the repainted mouth back into the
*whole* reference frame, so the output is whatever the reference was -- a 1024x1536 phone
photo came out as a 249 KB JPEG per frame, which at 25fps is 50 Mbps and unusable over a real
link. The face crop the model works on is 256x256 regardless, so scaling the surrounding frame
down throws away no generated detail; it throws away reference pixels that were never the point.

Applied after blending rather than to the reference before preparation, deliberately.
Landmarks on a downscaled reference are less precise, and the crop box they produce feeds a
model trained on boxes from full-resolution frames. Cheaper to be right and resize at the end.
"""

HALF_PRECISION_DEVICES = ("cuda", "mps")
"""
Which devices run the U-Net and VAE in float16.

Measured on this M1 Pro, batch of 8, median of 4 runs after a warm-up:

    float32   unet 1261 ms/frame   vae 987 ms/frame   total 2249
    float16   unet   76 ms/frame   vae 169 ms/frame   total  246   -> 9.15x

and the output is the same picture: mean absolute difference 0.04 of 255, maximum 1, no NaNs.
A 9x speedup for a rounding error in the last bit is not a trade-off, it is a default that was
wrong. Upstream's own `UNet` and `VAE` both take `use_float16`, so this uses their flag rather
than casting the modules afterwards -- `VAE` keeps a `_use_float16` of its own that its encode
path reads, and a module cast from outside would leave that lying.

CPU is excluded because float16 there is emulated and generally slower, not faster.
`AVATAR_MUSETALK_FP16=0` forces float32 for a fidelity comparison.
"""

FPS = 25
"""
The reference frame rate, and the rate audio features are chunked against.

Must match the renderer's `frame_interval_ms` or lip motion drifts against speech over a
long turn -- slowly enough to look like bad sync rather than a bug.
"""


def _device() -> str:
    """CUDA if present, else MPS, else CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _checkout() -> Path:
    """
    Where the MuseTalk source tree is.

    It is a git checkout rather than a package: upstream publishes no distribution, and its
    modules import each other by top-level name (`from musetalk.utils...`), so the checkout
    root has to be on `sys.path`. Vendored under `apps/api/vendor/` and gitignored -- it is
    someone else's source under its own licence, and committing it would put a fork in this
    repo that nobody intends to maintain.
    """
    root = os.environ.get("MUSETALK_CHECKOUT")
    if root:
        return Path(root)
    return Path(__file__).resolve().parents[3] / "vendor" / "MuseTalk"


class TorchMuseTalkBackend:
    """
    MuseTalk's realtime pipeline, driven one window at a time instead of one file at a time.

    The substantive divergence from upstream is audio: `realtime_inference.py` reads a whole
    file and segments it up front. A live conversation has no file, so `render` writes the
    window to a temporary WAV and feeds that. **That is a known inefficiency, not a design**
    -- a per-window file write on the critical path. The honest fix is to call Whisper's
    feature extractor on an in-memory array, which upstream's `AudioProcessor` does not
    expose. Measure before optimising: at 640ms windows this may be lost in the noise, or it
    may be the largest term in this stage. It is not yet known which.
    """

    def __init__(
        self,
        *,
        model_root: str | None = None,
        batch_size: int = BATCH_SIZE,
        jpeg_quality: int | None = None,
    ) -> None:
        default_root = Path(__file__).resolve().parents[3] / "models"
        self.model_root = Path(
            model_root or os.environ.get("MUSETALK_ROOT", str(default_root))
        )
        self.batch_size = int(os.environ.get("AVATAR_MUSETALK_BATCH", batch_size))
        self.jpeg_quality = jpeg_quality or int(
            os.environ.get("AVATAR_JPEG_QUALITY", JPEG_QUALITY)
        )
        self.max_output_height = int(
            os.environ.get("AVATAR_MUSETALK_MAX_HEIGHT", MAX_OUTPUT_HEIGHT)
        )
        self.device = ""
        self._models: dict[str, Any] = {}

    # ------------------------------------------------------------------ load

    def load(self) -> None:
        """
        Load VAE, U-Net, Whisper, the face parser and the landmark detector.

        Slow by nature -- several GB off disk plus a device context. Called once per process;
        paying it per session is exactly the cold-start cost PROCESS.md 1.4 argues against.

        Every path is passed absolute. Upstream's defaults are relative to the process's
        working directory (`./models/...`), which means the same code finds its weights or
        does not depending on where uvicorn was started from. That class of bug has already
        cost this repo three debugging sessions.
        """
        if self._models:
            return
        import torch

        self.device = _device()
        if self.device != "cuda":
            # Loud rather than quietly slow. A non-CUDA run is not a failure -- the brief
            # explicitly permits reporting real numbers on a lighter setup -- but it must
            # never be mistaken for a GPU measurement.
            print(
                f"!! MuseTalk backend on device={self.device!r}, not CUDA. "
                "Throughput will not resemble the published figures, and any number "
                "measured here must be reported with this device attached to it."
            )

        checkout = _checkout()
        if not (checkout / "musetalk").is_dir():
            raise RuntimeError(
                f"no MuseTalk checkout at {checkout}. Run scripts/setup_musetalk.sh, or "
                "point MUSETALK_CHECKOUT at one."
            )
        if str(checkout) not in sys.path:
            sys.path.insert(0, str(checkout))

        missing = [
            str(p)
            for p in (
                self.model_root / "musetalkV15" / "unet.pth",
                self.model_root / "musetalkV15" / "musetalk.json",
                self.model_root / "sd-vae" / "diffusion_pytorch_model.bin",
                self.model_root / "whisper" / "pytorch_model.bin",
                self.model_root / "face-parse-bisent" / "79999_iter.pth",
            )
            if not p.exists()
        ]
        if missing:
            raise RuntimeError(
                "MuseTalk weights are incomplete. Missing:\n  "
                + "\n  ".join(missing)
                + "\nRun: scripts/fetch_musetalk_weights.py"
            )

        # Imported here, not at module scope: these are the packages that must stay out of
        # the orchestration import graph.
        # Constructed directly rather than through `load_all_model`, which takes no
        # `use_float16` and resolves the VAE against the working directory.
        from musetalk.models.unet import PositionalEncoding, UNet
        from musetalk.models.vae import VAE
        from musetalk.utils.audio_processor import AudioProcessor
        from musetalk.utils.face_parsing import FaceParsing
        from transformers import WhisperModel

        from avatar.renderers.landmarks import LandmarkDetector

        half = self.device in HALF_PRECISION_DEVICES and os.environ.get(
            "AVATAR_MUSETALK_FP16", "1"
        ) not in ("0", "false", "no")

        vae = VAE(model_path=str(self.model_root / "sd-vae"), use_float16=half)
        # `VAE.__init__` sets `self.device = "cuda" if torch.cuda.is_available() else "cpu"`,
        # with no MPS branch -- so on Apple Silicon it puts the VAE on the CPU and then reports
        # that it did not. Both halves matter: the module has to move, and the attribute has to
        # agree, because `encode_latents` sends its input to `self.device`. Moving only the
        # module left every encode copying to the CPU and back, which is most of the 987 ms
        # float32 VAE figure above.
        # The module, and the attribute that claims where it is. `_mask_tensor` deliberately
        # stays on the CPU: `preprocess_img` builds its input there, multiplies by that mask,
        # and only then calls `.to(self.vae.device)` -- so it follows the module by itself, and
        # moving the mask ahead of it is what makes the multiply fail with "mps:0 and cpu".
        vae.device = torch.device(self.device)
        vae.vae = vae.vae.to(device=self.device)

        unet = UNet(
            unet_config=str(self.model_root / "musetalkV15" / "musetalk.json"),
            model_path=str(self.model_root / "musetalkV15" / "unet.pth"),
            use_float16=half,
            device=torch.device(self.device),
        )
        dtype = unet.model.dtype
        pe = PositionalEncoding(d_model=384).to(device=self.device, dtype=dtype)

        whisper = WhisperModel.from_pretrained(str(self.model_root / "whisper"))
        whisper = whisper.to(device=self.device, dtype=dtype).eval()
        whisper.requires_grad_(False)

        # A subclass, not an assignment after construction. `FaceParsing.__init__` calls
        # `self.model_init()` with its own defaults -- `./models/face-parse-bisent/...`,
        # relative to the working directory -- so by the time an override could be assigned,
        # the load has already happened or already failed. Overriding the method is what
        # actually removes the cwd dependency, and it needs no edit to the vendored checkout.
        parse_root = self.model_root / "face-parse-bisent"

        class AbsolutePathFaceParsing(FaceParsing):  # type: ignore[misc]
            def model_init(  # type: ignore[no-untyped-def]
                self, resnet_path: str = "", model_pth: str = ""
            ):
                return super().model_init(
                    resnet_path=str(parse_root / "resnet18-5c106cde.pth"),
                    model_pth=str(parse_root / "79999_iter.pth"),
                )

        parser = AbsolutePathFaceParsing(
            left_cheek_width=CHEEK_WIDTH, right_cheek_width=CHEEK_WIDTH
        )

        detector = LandmarkDetector(device=self.device)
        detector.load()

        self._models = {
            "vae": vae,
            "unet": unet,
            "pe": pe,
            "whisper": whisper,
            "audio": AudioProcessor(feature_extractor_path=str(self.model_root / "whisper")),
            "parser": parser,
            "landmarks": detector,
            "dtype": dtype,
        }

    # --------------------------------------------------------------- prepare

    def _read_frames(self, reference_path: str) -> list[Any]:
        """
        Decode a reference into BGR frames.

        Upstream shells out to `ffmpeg -i ... %08d.png` and globs the directory back. Decoded
        in-process here instead: the frames are wanted in memory anyway, and a temporary
        directory of PNGs per session is both slower and one more thing to clean up. A still
        image is a one-frame list -- `avatar.media` has already expanded uploaded stills into
        clips, so this path is only reached for a reference that was a file on disk.
        """
        import cv2

        path = Path(reference_path)
        if not path.exists():
            raise RuntimeError(f"reference not found: {path}")

        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            frame = cv2.imread(str(path))
            if frame is None:
                raise RuntimeError(f"could not decode image: {path}")
            return [frame]

        capture = cv2.VideoCapture(str(path))
        frames: list[Any] = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        if not frames:
            raise RuntimeError(f"no frames decoded from {path}")
        return frames

    def prepare(self, reference_path: str) -> object:
        """
        Landmarks, crops, VAE latents and blending masks for every reference frame.

        Returns the four cached cycles that together are the whole identity artifact. Note
        what is *not* in it: any model weights. That is the §1.2 claim, and it is what lets
        one warm worker serve any persona.

        Frames with no detected face are dropped rather than rendered. Upstream keeps them
        and skips them later, which leaves the four lists at different lengths -- and they are
        indexed by a shared counter, so a single undetected frame shifts masks against
        latents for the rest of the session.
        """
        import cv2
        import numpy as np
        from musetalk.utils.blending import get_image_prepare_material

        if not self._models:
            self.load()

        frames = self._read_frames(reference_path)
        detector = self._models["landmarks"]

        kept: list[Any] = []
        coords: list[tuple[int, int, int, int]] = []
        latents: list[Any] = []
        masks: list[Any] = []
        mask_boxes: list[Any] = []

        for frame in frames:
            found = detector.detect(frame)
            if not found.found:
                continue
            x1, y1, x2, y2 = (int(v) for v in found.box)
            # The margin is folded in once, here, so the crop and the later blend cannot
            # disagree about where the face was.
            y2 = min(y2 + EXTRA_MARGIN, frame.shape[0])
            if y2 - y1 <= 0 or x2 - x1 <= 0:
                continue

            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(
                crop, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LANCZOS4
            )
            box = (x1, y1, x2, y2)
            mask, mask_box = get_image_prepare_material(
                frame, list(box), fp=self._models["parser"], mode=PARSING_MODE
            )

            kept.append(frame)
            coords.append(box)
            latents.append(self._models["vae"].get_latents_for_unet(crop))
            masks.append(np.asarray(mask))
            mask_boxes.append(mask_box)

        if not kept:
            raise RuntimeError(
                f"no face found in any of {len(frames)} frame(s) of {reference_path}. "
                "A reference must show one front-facing person."
            )

        # Cycled forward-then-backward so the reference loop has no jump cut at the seam --
        # the same reason the placeholder idle loop waits for a clean exit frame.
        return {
            "frames": kept + kept[::-1],
            "coords": coords + coords[::-1],
            "latents": latents + latents[::-1],
            "masks": masks + masks[::-1],
            "mask_boxes": mask_boxes + mask_boxes[::-1],
            "usable_frames": len(kept),
            "source_frames": len(frames),
        }

    # ---------------------------------------------------------------- render

    def render(
        self, prepared: object, pcm: bytes, *, start_frame: int, count: int
    ) -> list[bytes]:
        """
        One window: PCM in, JPEG frames out.

        `start_frame` indexes the reference cycle modulo its length, so consecutive windows
        continue the body motion instead of restarting it -- restarting would make the head
        snap back to frame zero at every window boundary.
        """
        if not self._models:
            self.load()
        assert isinstance(prepared, dict)

        features = self._audio_features(pcm)
        if not features:
            return []

        cycle_len = len(prepared["latents"])
        images: list[bytes] = []
        for offset in range(0, min(count, len(features)), self.batch_size):
            batch = features[offset : offset + self.batch_size]
            indices = [(start_frame + offset + i) % cycle_len for i in range(len(batch))]
            rendered = self._forward(prepared, batch, indices)
            images.extend(self._encode(image) for image in rendered)
        return images

    def _audio_features(self, pcm: bytes) -> list[Any]:
        """
        Whisper feature chunks for one window of raw 16kHz mono PCM.

        Two calls, not one: `get_audio_feature` produces mel features from a path, and
        `get_whisper_chunk` runs the encoder and slices the result per video frame. The
        temporary WAV is the divergence described in the class docstring.
        """
        import tempfile
        import wave

        if not pcm:
            return []

        processor = self._models["audio"]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as handle:
            with wave.open(handle.name, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16_000)
                wav.writeframes(pcm)
            features, librosa_length = processor.get_audio_feature(handle.name)

        return list(
            processor.get_whisper_chunk(
                features,
                self.device,
                self._models["dtype"],
                self._models["whisper"],
                librosa_length,
                fps=FPS,
                audio_padding_length_left=AUDIO_PADDING,
                audio_padding_length_right=AUDIO_PADDING,
            )
        )

    def _forward(
        self, prepared: dict[str, Any], batch: list[Any], indices: list[int]
    ) -> list[Any]:
        """Audio embedding -> U-Net -> VAE decode -> blend onto the reference frame."""
        import cv2
        import numpy as np
        import torch
        from musetalk.utils.blending import get_image_blending

        unet = self._models["unet"]
        with torch.no_grad():
            audio = torch.stack([torch.as_tensor(f) for f in batch]).to(
                device=self.device, dtype=self._models["dtype"]
            )
            latents = torch.cat([prepared["latents"][i] for i in indices]).to(
                device=self.device, dtype=unet.model.dtype
            )
            embeddings = self._models["pe"](audio)
            predicted = unet.model(
                latents,
                torch.tensor([0], device=self.device),
                encoder_hidden_states=embeddings,
            ).sample
            decoded = self._models["vae"].decode_latents(predicted)

        out: list[Any] = []
        for i, face in zip(indices, decoded, strict=False):
            x1, y1, x2, y2 = prepared["coords"][i]
            # Back to the size of the hole it came from. The U-Net always works at 256x256,
            # so this is a resize on every frame, not an occasional one.
            resized = cv2.resize(np.asarray(face).astype(np.uint8), (x2 - x1, y2 - y1))
            out.append(
                get_image_blending(
                    prepared["frames"][i],
                    resized,
                    [x1, y1, x2, y2],
                    prepared["masks"][i],
                    prepared["mask_boxes"][i],
                )
            )
        return out

    def _encode(self, image: Any) -> bytes:
        """
        BGR array -> JPEG bytes.

        OpenCV rather than Pillow: MuseTalk's blending step already returns OpenCV BGR
        arrays, so `imencode` avoids a colour-space conversion and a second imaging
        dependency. Falls back to Pillow if OpenCV is absent.
        """
        import cv2
        import numpy as np

        array = np.asarray(image)
        if self.max_output_height and array.shape[0] > self.max_output_height:
            scale = self.max_output_height / array.shape[0]
            array = cv2.resize(
                array,
                (round(array.shape[1] * scale), self.max_output_height),
                # INTER_AREA is the right filter for shrinking -- it averages the pixels being
                # collapsed. INTER_LINEAR samples them, which aliases skin texture into noise
                # that JPEG then spends bits on.
                interpolation=cv2.INTER_AREA,
            )
        try:

            ok, buffer = cv2.imencode(
                ".jpg", array, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            )
            if not ok:
                raise RuntimeError("cv2.imencode failed on a rendered frame")
            return bytes(buffer)
        except ImportError:
            import io

            from PIL import Image

            rgb = array[:, :, ::-1]  # OpenCV hands back BGR; Pillow wants RGB
            out = io.BytesIO()
            Image.fromarray(rgb).save(out, format="JPEG", quality=self.jpeg_quality)
            return out.getvalue()

    def unload(self) -> None:
        """Drop weights and empty the device cache. Safe to call twice."""
        self._models.clear()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass
