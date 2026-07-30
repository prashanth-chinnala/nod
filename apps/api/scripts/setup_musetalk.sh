#!/usr/bin/env bash
#
# Set up the real renderer: a Python 3.10 environment, MuseTalk's source, and its weights.
#
# `set -euo pipefail` on the first line, and that is the entire point of this file existing.
# Upstream's `download_weights.sh` has no `set -e` and ends with an unconditional
# "✅ All weights have been downloaded successfully!" -- which is how the first spike run
# finished with 96 MB on disk, exit code 0, and a success message. Every step here either
# works or stops the script.
#
# Idempotent: re-running skips what is already present and re-verifies it.
#
#   ./scripts/setup_musetalk.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=".venv-musetalk"
CHECKOUT="vendor/MuseTalk"
PY310="${PY310:-/opt/homebrew/bin/python3.10}"

say() { printf '\n=== %s\n' "$1"; }

# ---------------------------------------------------------------- interpreter
#
# 3.10, not the 3.12 the runtime uses, and a separate environment rather than the runtime's.
# Two hard constraints force it:
#
#   * numpy must be < 2. `opencv-python` is compiled against the numpy 1.x C ABI and aborts
#     against numpy 2 with `AttributeError: _ARRAY_API not found` -- an import failure, not a
#     warning. torch pulls in numpy 2 unless pinned.
#   * MuseTalk pins `transformers==4.39.2`, `diffusers==0.30.2`, `huggingface_hub==0.30.2`,
#     which the console assistant's LangChain stack would fight over.
#
# The runtime's own package declares `dependencies = []` and CI runs it with no GPU and no
# weights. Merging these two dependency sets would spend that guarantee to serve one renderer.
say "interpreter"
if [ ! -x "$PY310" ]; then
  echo "!! no python3.10 at $PY310"
  echo "   brew install python@3.10, or set PY310=/path/to/python3.10"
  exit 1
fi
if [ ! -d "$VENV" ]; then
  "$PY310" -m venv "$VENV"
fi
"$VENV/bin/pip" install -q -U pip wheel setuptools
"$VENV/bin/python" -V

# ------------------------------------------------------------------- checkout
#
# A git clone, not a pip install: upstream publishes no distribution, and its modules import
# each other by top-level name, so the tree has to sit on sys.path. Gitignored -- someone
# else's source under its own licence has no business being committed here.
say "MuseTalk source"
if [ ! -d "$CHECKOUT/musetalk" ]; then
  git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git "$CHECKOUT"
else
  echo "already at $CHECKOUT"
fi

# --------------------------------------------------------------- dependencies
say "dependencies"
"$VENV/bin/pip" install -q torch torchvision torchaudio

# Upstream's requirements.txt, minus two entries and plus one.
#
# Dropped: `tensorflow==2.12.0` and `tensorboard==2.12.0`. Nothing in the repository imports
# either -- checked with grep across every .py -- and tensorflow 2.12 has no macOS arm64 wheel,
# so on this machine they are a hard install failure in service of dead weight.
#
# Dropped: `mmcv==2.0.1` / `mmpose==1.1.0` (installed via openmim in upstream's README). They
# exist for one line, `keypoints[0][23:91]`, and `avatar.renderers.landmarks` produces the same
# iBUG-68 points from a pure-torch detector. They are also the reason the first spike run died:
# pinned, compiled, Linux/Windows wheels only, against an old torch.
#
# Added: `face-alignment`, which is what replaces them.
#
# `numpy<2` last, deliberately: torch's install pulls numpy 2 in, and this pins it back down
# before anything imports cv2.
"$VENV/bin/pip" install -q \
  diffusers==0.30.2 \
  accelerate==0.28.0 \
  opencv-python==4.9.0.80 \
  soundfile==0.12.1 \
  transformers==4.39.2 \
  huggingface_hub==0.30.2 \
  librosa==0.11.0 \
  einops==0.8.1 \
  omegaconf \
  ffmpeg-python \
  "imageio[ffmpeg]" \
  face-alignment
"$VENV/bin/pip" install -q "numpy<2"

if ! command -v ffmpeg >/dev/null; then
  echo "!! ffmpeg is missing. MuseTalk needs it at inference time and so does avatar.media."
  echo "   brew install ffmpeg"
  exit 1
fi

# ------------------------------------------------------------------- weights
say "weights (3.7 GB, verified on arrival)"
"$VENV/bin/python" scripts/fetch_musetalk_weights.py

# -------------------------------------------------------------------- proof
#
# Importing is not evidence. This loads every model onto the device and reports which one it
# got, because a silent fall back to CPU is the difference between a demo and a slideshow.
say "checking the stack"
PYTHONPATH=src "$VENV/bin/python" - <<'PYCHECK'
import warnings
warnings.filterwarnings("ignore")
import numpy, torch, cv2
print(f"numpy {numpy.__version__}   torch {torch.__version__}   cv2 {cv2.__version__}")
print(f"cuda {torch.cuda.is_available()}   mps {torch.backends.mps.is_available()}")

from avatar.renderers.musetalk_torch import TorchMuseTalkBackend
backend = TorchMuseTalkBackend()
backend.load()
print(f"all models loaded on device={backend.device!r}")
PYCHECK

say "done"
cat <<'NEXT'
The renderer is installed and its models load. To use it in a session:

    AVATAR_RENDERER=musetalk uvicorn avatar.server:app

but read this first, because on Apple Silicon it will not keep up: measured 2604 ms per frame
(median of 6 batches) against a 40 ms budget at 25fps. Correct output, 65x too slow. The
numbers and what they mean are in ROADMAP.md.
NEXT
