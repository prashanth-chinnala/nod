"""
The GPU half of the MuseTalk renderer. Every torch, CUDA, and OpenCV call lives here.

Separated from `musetalk.py` so that the streaming logic -- buffering, windowing, epoch
tagging, reset -- is testable on a machine with no GPU, and so that `import avatar` never
pulls torch into the graph. `tests/test_boundaries.py` enforces the second part.

**This file has never been executed.** There is no CUDA device in the development
environment and MuseTalk's documented stack does not install on the current free Colab
runtime (see PROCESS.md 2.2.1). It is written against MuseTalk's published realtime
inference script and its README, and the parts most likely to need adjustment are marked.
Treat every line as unverified until a spike run says otherwise -- and nothing here produces
a number that may be quoted as a measurement.

**Why JPEG here and PNG in the stub.** The stub draws flat colour blocks, where lossless PNG
is both smaller and artifact-free. A rendered face is photographic, where PNG barely
compresses -- a 256x256 photographic PNG runs 60-120 KB, which at 25fps is 12-24 Mbps and
puts us straight back into the bandwidth wall that PNG was introduced to solve. JPEG at
quality 82 lands the same frame around 8-15 KB, roughly 2-3 Mbps. The client sniffs the
format from magic bytes, so this needs no protocol change and no coordinated deploy.
"""

from __future__ import annotations

import os
from typing import Any

JPEG_QUALITY = 82
"""
Quality/size trade-off for rendered frames.

82 is the usual knee: visible ringing starts below ~75, and above ~90 the file grows fast
for detail a 256x256 face region does not carry. `AVATAR_JPEG_QUALITY` overrides it, because
the right value depends on the link and on the model's output resolution, neither of which
is measured yet.
"""

BATCH_SIZE = 8
"""
Frames per U-Net forward pass.

Larger batches use the GPU better and cost first-frame latency, since no frame in a batch
emits until all of them are done. 8 is MuseTalk's own realtime default. `NOT YET MEASURED`
on a T4 -- tune once, with numbers, rather than guessing repeatedly.
"""


def _device() -> str:
    """CUDA if present, else MPS, else CPU -- with the honest caveat about the last two."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon runs the VAE and U-Net but MuseTalk's face pipeline depends on
    # OpenMMLab packages with compiled CUDA ops, so MPS is unlikely to work end to end
    # without substituting the detector. Reported rather than silently degraded.
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TorchMuseTalkBackend:
    """
    MuseTalk's realtime pipeline, driven one window at a time instead of one file at a time.

    The mapping onto MuseTalk's own script, so the divergence is auditable:

    | MuseTalk's `realtime_inference.py` | Here |
    |---|---|
    | `load_all_model()` | `load()` |
    | `Avatar.prepare_material()` | `prepare()` |
    | `get_audio_feature(audio_path)` | `render()`, but from a PCM buffer, not a path |
    | `datagen()` + the inference loop | `render()` |
    | `process_frames()` / `get_image_blending()` | `render()`, then JPEG encode |

    The one substantive divergence is audio: upstream reads a file and segments it up front.
    A live conversation has no file, so `render` writes the window to a temporary WAV and
    feeds that. **That is a known inefficiency, not a design** -- a per-window file write on
    the critical path. The honest fix is to call Whisper's feature extractor on an in-memory
    array directly, which the upstream API does not currently expose. Measure before
    optimising: at 640ms windows this may be lost in the noise, or it may be the largest term
    in this stage. It is not yet known which.
    """

    def __init__(
        self,
        *,
        model_root: str | None = None,
        batch_size: int = BATCH_SIZE,
        jpeg_quality: int | None = None,
    ) -> None:
        self.model_root = model_root or os.environ.get("MUSETALK_ROOT", "models")
        self.batch_size = batch_size
        self.jpeg_quality = jpeg_quality or int(
            os.environ.get("AVATAR_JPEG_QUALITY", JPEG_QUALITY)
        )
        self.device = ""
        self._models: dict[str, Any] = {}

    def load(self) -> None:
        """
        Load VAE, U-Net, and the Whisper audio processor onto the device.

        Slow by nature -- several GB off disk plus a CUDA context. Called once per process;
        paying it per session is exactly the cold-start cost PROCESS.md 1.4 argues against.
        """
        if self._models:
            return
        self.device = _device()
        if self.device != "cuda":
            # Loud rather than quietly slow. A CPU run is not a failure -- the brief
            # explicitly permits reporting real numbers on a lighter setup -- but it must
            # never be mistaken for a GPU measurement.
            print(
                f"!! MuseTalk backend on device={self.device!r}, not CUDA. "
                "Throughput will not resemble the published figures, and any number "
                "measured here must be reported with this device attached to it."
            )

        # Imported here, not at module scope: these are the packages that must stay out of
        # the orchestration import graph.
        from musetalk.utils.utils import load_all_model

        vae, unet, pe = load_all_model()
        self._models = {"vae": vae, "unet": unet, "pe": pe}

    def prepare(self, reference_path: str) -> object:
        """
        Face detection, parsing, and VAE encoding of the reference frames.

        Returns MuseTalk's four cached cycles -- coordinates, latents, masks, mask
        coordinates -- which together are the whole identity artifact. Note what is *not*
        in it: any model weights. That is the §1.2 claim, and it is what lets one warm
        worker serve any persona.
        """
        if not self._models:
            self.load()

        from musetalk.utils.blending import (
            get_image_prepare_material,
        )
        from musetalk.utils.preprocessing import (
            get_landmark_and_bbox,
            read_imgs,
        )

        frames = read_imgs(reference_path)
        coords, landmarks = get_landmark_and_bbox(frames, bbox_shift=0)
        latents = self._models["vae"].get_latents_for_unet(frames, coords)
        masks = [get_image_prepare_material(f, c) for f, c in zip(frames, coords, strict=False)]

        # Cycled forward-then-backward so the reference loop has no jump cut at the seam --
        # the same reason the placeholder idle loop waits for a clean exit frame.
        return {
            "frames": frames + frames[::-1],
            "coords": coords + coords[::-1],
            "latents": latents + latents[::-1],
            "masks": masks + masks[::-1],
            "landmarks": landmarks,
        }

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
        Whisper features for one window of raw PCM.

        The temporary WAV is the divergence described in the class docstring. It is here
        because MuseTalk's `get_audio_feature` takes a path; it should be replaced the
        moment an in-memory entry point exists, and it should be measured before then.
        """
        import tempfile
        import wave

        from musetalk.utils.audio_processor import (
            AudioProcessor,
        )

        processor = self._models.setdefault("audio", AudioProcessor())
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as handle:
            with wave.open(handle.name, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16_000)
                wav.writeframes(pcm)
            whisper_features = processor.get_audio_feature(handle.name)
        return list(processor.get_whisper_chunk(whisper_features))

    def _forward(
        self, prepared: dict[str, Any], batch: list[Any], indices: list[int]
    ) -> list[Any]:
        """Audio embedding -> U-Net -> VAE decode -> blend onto the reference frame."""
        import torch
        from musetalk.utils.blending import (
            get_image_blending,
        )

        with torch.no_grad():
            audio = torch.stack([torch.as_tensor(f) for f in batch]).to(self.device)
            latents = torch.stack([prepared["latents"][i] for i in indices]).to(self.device)
            embeddings = self._models["pe"](audio)
            predicted = (
                self._models["unet"]
                .model(
                    latents,
                    torch.tensor([0], device=self.device),
                    encoder_hidden_states=embeddings,
                )
                .sample
            )
            decoded = self._models["vae"].decode_latents(predicted)

        return [
            get_image_blending(
                prepared["frames"][i], face, prepared["coords"][i], prepared["masks"][i]
            )
            for i, face in zip(indices, decoded, strict=False)
        ]

    def _encode(self, image: Any) -> bytes:
        """
        BGR array -> JPEG bytes.

        OpenCV rather than Pillow: MuseTalk's blending step already returns OpenCV BGR
        arrays, so `imencode` avoids a colour-space conversion and a second imaging
        dependency. Falls back to Pillow if OpenCV is absent.
        """
        try:
            import cv2

            ok, buffer = cv2.imencode(
                ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            )
            if not ok:
                raise RuntimeError("cv2.imencode failed on a rendered frame")
            return bytes(buffer)
        except ImportError:
            import io

            from PIL import Image

            rgb = image[:, :, ::-1]  # OpenCV hands back BGR; Pillow wants RGB
            out = io.BytesIO()
            Image.fromarray(rgb).save(out, format="JPEG", quality=self.jpeg_quality)
            return out.getvalue()

    def unload(self) -> None:
        """Drop weights and empty the CUDA cache. Safe to call twice."""
        self._models.clear()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
