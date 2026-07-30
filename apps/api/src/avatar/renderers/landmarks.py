"""
Face landmarks and the MuseTalk crop box, without OpenMMLab.

**What this replaces and why.** MuseTalk's `musetalk/utils/preprocessing.py` opens with three
module-scope side effects: it builds an mmpose RTMPose model, loads `dw-ll_ucoco_384.pth`, and
constructs an S3FD detector — at import, before any caller has asked for anything. The mmpose
half is the problem. It arrives via `openmim` as `mmcv==2.0.1` + `mmpose==1.1.0`, which are
pinned, compiled, published as wheels for Linux and Windows only, and pinned against a torch
generation several releases old. That stack is what the first spike run died on, and on Apple
Silicon it does not build at all.

What it is used *for* turns out to be one line:

    face_land_mark = keypoints[0][23:91]

Indices 23:91 of the COCO-WholeBody 133-point skeleton are its 68 face keypoints, and that is
the iBUG-68 layout — the same layout `face_alignment` has produced since 2017 from a pure-torch
FAN, with no compiled ops, on CUDA, MPS or CPU. The bounding box upstream pairs it with already
comes from `face_detection`, which is S3FD vendored *from face_alignment*. So the substitution
replaces half a file's worth of dependency with the library the other half was already copied
from.

**The arithmetic below is upstream's, deliberately unchanged.** Points 28/29/30 are the nose
bridge; 29 is the mid-face anchor; the box runs from the widest landmark on each side, down to
the lowest, and up from the anchor by the same distance as the anchor sits above the chin —
which is what makes the crop cover the mouth with headroom rather than hug the jaw. Every
constant, the `max(0, ...)` floor, and the degenerate-box fallback are copied rather than
improved, because the U-Net was trained on crops this function produced. A "better" box is a
differently-distributed input to a fixed model.

**What is honestly different.** The landmarks come from a different detector, so they are not
bit-identical to RTMPose's, so the crop can differ by a pixel or two, so output quality may
differ from MuseTalk's published samples. That is a real caveat and it is not measured. It is
also the only way this runs on the development machine at all — the alternative was no face.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

NO_FACE = (0.0, 0.0, 0.0, 0.0)
"""
Upstream's `coord_placeholder`, kept identical.

Its callers test `bbox == coord_placeholder` to skip a frame, so the value and its float type
are load-bearing: `(0, 0, 0, 0)` would compare equal in Python but would not survive a round
trip through the pickle upstream writes.
"""

DETECT_MAX_SIDE = 640
"""
Longest side the detector sees, in pixels.

Not an optimisation — a correction. S3FD on a 1536x1024 phone frame measured 5.4s per image
here, and a reference clip is hundreds of frames, so preparation would take longer than the
interview. Faces are found at 640 and the landmarks are scaled back up, which costs sub-pixel
precision on a box that is then rounded to integers anyway.

Set `AVATAR_LANDMARK_MAX_SIDE=0` to disable and detect at full resolution.
"""


@dataclass(frozen=True)
class FaceBox:
    """A crop box, plus the landmarks it came from, so a caller can check the fit."""

    box: tuple[float, float, float, float]
    landmarks: Any | None
    """68x2 int32, or None when no face was found."""

    @property
    def found(self) -> bool:
        return self.box != NO_FACE


class LandmarkDetector:
    """
    68-point landmarks and MuseTalk's crop box.

    Built lazily and held for the process. Loading FAN plus S3FD measured 2.1s on MPS and
    0.6s on CPU here, which is cheap once and absurd per frame.
    """

    def __init__(self, *, device: str = "", max_side: int | None = None) -> None:
        self.device = device
        self.max_side = (
            max_side
            if max_side is not None
            else int(os.environ.get("AVATAR_LANDMARK_MAX_SIDE", DETECT_MAX_SIDE))
        )
        self._fa: Any = None

    def load(self) -> None:
        if self._fa is not None:
            return
        import face_alignment
        import torch

        if not self.device:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        self._fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device=self.device,
            # Upstream passes `flip_input=False`; matched so the landmarks are produced the
            # same way rather than averaged over a mirrored pass.
            flip_input=False,
            # `torch.compile` cost 77s on the first call here and inductor warns that MPS has
            # "not enough SMs" — it is reasoning about a GPU that is not there. A minute of
            # compilation to prepare one face is not a trade worth making.
            compile=False,
        )

    def detect(self, frame: Any) -> FaceBox:
        """
        One BGR frame in (OpenCV order), one crop box out.

        BGR because every caller here is an OpenCV reader, and `face_alignment` wants RGB —
        converted in one place rather than at each call site, which is where a channel swap
        silently costs accuracy without failing.
        """
        import cv2
        import numpy as np

        self.load()
        height, width = frame.shape[:2]

        scale = 1.0
        detect_on = frame
        if self.max_side and max(height, width) > self.max_side:
            scale = self.max_side / max(height, width)
            detect_on = cv2.resize(
                frame,
                (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_AREA,
            )

        rgb = np.ascontiguousarray(detect_on[:, :, ::-1])
        found = self._fa.get_landmarks_from_image(rgb, return_bboxes=True)
        landmark_sets, _scores, boxes = found if found is not None else (None, None, None)
        if not landmark_sets:
            return FaceBox(NO_FACE, None)

        # The largest face, when there is more than one. Upstream takes the detector's first,
        # which is by descending confidence -- but in an interview reference the subject is the
        # near, large face, and a confident face in the background is still the wrong one.
        index = 0
        if boxes is not None and len(boxes) > 1:
            index = max(
                range(len(boxes)),
                key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]),
            )

        landmarks = np.asarray(landmark_sets[index], dtype=np.float64)
        if scale != 1.0:
            landmarks = landmarks / scale
        landmarks = landmarks.astype(np.int32)

        detector_box: tuple[float, float, float, float] | None = None
        if boxes is not None and len(boxes) > index:
            raw = np.asarray(boxes[index], dtype=np.float64)[:4] / scale
            detector_box = (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))

        return FaceBox(self._crop_box(landmarks, detector_box), landmarks)

    @staticmethod
    def _crop_box(
        landmarks: Any, detector_box: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float]:
        """
        MuseTalk's box, from `get_landmark_and_bbox`, transcribed.

        `bbox_shift` is intentionally absent. Upstream exposes it as a manual nudge an operator
        tunes by watching output, and there is nobody watching during a live session; v1.5's
        own inference script hardcodes it to 0 for the same reason.
        """
        import numpy as np

        anchor = landmarks[29]
        chin = int(np.max(landmarks[:, 1]))
        # How far the anchor sits above the lowest landmark, mirrored upwards for headroom.
        below = chin - int(anchor[1])
        upper = max(0, int(anchor[1]) - below)

        box = (
            float(np.min(landmarks[:, 0])),
            float(int(upper)),
            float(np.max(landmarks[:, 0])),
            float(chin),
        )
        x1, y1, x2, y2 = box
        if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
            # Upstream falls back to the detector's box here. Without one, refuse the frame
            # rather than return a degenerate crop -- a zero-width slice reaches the VAE as an
            # empty tensor and fails several layers away from the cause.
            return detector_box if detector_box is not None else NO_FACE
        return box


def get_landmark_and_bbox(
    frames: list[Any], detector: LandmarkDetector | None = None
) -> tuple[list[tuple[float, float, float, float]], list[Any]]:
    """
    Drop-in for MuseTalk's function of the same name, minus the file reading.

    Upstream takes paths and reads them; every caller here already holds decoded frames, and
    writing them to disk to hand back a path was not worth matching. The return shape is
    upstream's: `(coords_list, frames)`, one box per frame, in order.
    """
    detector = detector or LandmarkDetector()
    return [detector.detect(frame).box for frame in frames], frames
