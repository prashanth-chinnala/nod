# How to accomplish M0

Read after run 1 failed. This supersedes the triage guesswork: MuseTalk's own
`download_weights.sh` and README have now been read, and the failure has three causes
rather than one. **One of them was a bug in my notebook.**

## What actually went wrong

### Cause 1 — my inference command was incomplete (my error)

`notebooks/m0_musetalk_spike.ipynb` ran:

```
python -m scripts.inference --inference_config configs/inference/test.yaml --result_dir ...
```

MuseTalk v1.5 requires four more arguments, per its README:

```
--unet_model_path models/musetalkV15/unet.pth \
--unet_config      models/musetalkV15/musetalk.json \
--version          v15 \
--ffmpeg_path      <path to ffmpeg>
```

Without `--version v15` it tries to load the v1.0 checkpoint layout. That alone is
`exit_code: 1`. The notebook told you to check §5 for the real invocation and I had put a
guess in the cell instead of reading it — apologies.

### Cause 2 — the weights download fails silently, by design

`download_weights.sh` sets **`HF_ENDPOINT` to a mirror** rather than Hugging Face itself,
then runs a sequence of `huggingface-cli download --include ...` calls, plus `gdown` for
one file from Google Drive and `curl` for the ResNet18 backbone.

Three ways that produces 96MB and exit code 0:

- The mirror is frequently unreachable or rate-limited from Colab. Each failed
  `huggingface-cli` call prints an error and returns, but the script has no `set -e`, so
  the shell exits 0 regardless.
- `--include` filters can match nothing and still succeed.
- `gdown` on a Google Drive link hits the interstitial quota page and writes an HTML file
  where a checkpoint should be.

**Nothing validates a single byte.** 96MB is roughly `resnet18` (45MB) plus small configs
— i.e. only the `curl` step worked.

### Cause 3 — Python 3.12 versus a Python 3.10 pinned stack

Your runtime reported **`python: 3.12.13`**. MuseTalk pins:

| | Pinned |
|---|---|
| Python | 3.10 |
| torch | 2.0.1 (cu118) |
| mmcv | 2.0.1 |
| mmdet | 3.1.0 |
| mmpose | 1.1.0 |

There are no prebuilt `mmcv==2.0.1` wheels for Python 3.12, and building it from source
needs a matching CUDA toolchain. This explains the other implausible number: **`pip install`
finished in 13 seconds** for a stack that normally takes minutes. It did not install the
OpenMMLab packages — it resolved what it could and moved on.

That mismatch is itself a §2.1 finding. A model whose documented stack no longer installs
on the current free-tier runtime has a real clean-clone problem, and that belongs in the
memo whichever model you pick.

## The plan

Two viable routes. Read both before starting — the choice depends on how much of your
one-day box you have left.

### Route A — pin Python 3.10 with condacolab (recommended)

Fixes cause 3 head-on and keeps you on MuseTalk's documented, tested stack. Costs one
runtime restart and ~10 minutes.

1. `pip install -q condacolab` then `condacolab.install()` — **the runtime restarts**,
   which is expected, not a crash.
2. Create a 3.10 environment, install the pinned torch and the OpenMMLab stack via
   `openmim`.
3. Download weights with `HF_ENDPOINT` **unset** so it talks to Hugging Face directly.
4. **Verify every checkpoint's size before running anything.** This is the step whose
   absence cost you run 1.
5. Run the v1.5 command with all four required arguments.

`notebooks/m0_musetalk_v2.ipynb` implements exactly this.

### Route B — accept the version drift

Stay on Python 3.12, install the newest `mmcv`/`mmpose` that have 3.12 wheels, and see
whether MuseTalk's code still runs against them. Faster to attempt, and it may fail on an
API change inside `mmpose` that you would then be debugging instead of measuring.

Worth 30 minutes, not worth 3 hours. If it fails, take Route A.

### If both routes stall

The fallbacks from `M0_SPIKE.md` §5 are unchanged and all still legitimate:

- **Try Ditto.** Apache-2.0, streaming-native. Its TensorRT requirement fights an
  ephemeral runtime, which is a documented cost rather than a surprise.
- **Drop the resolution** and report the real number at that resolution.
- **Report the dead end.** "MuseTalk's documented stack does not install on the current
  free Colab runtime, and its weight downloader exits 0 having fetched 2% of the
  checkpoints" is a *finding*, not a failure to produce one. §2.1's setup-fragility
  criterion is asking for precisely this.

## What run 2 must produce

The bar is unchanged and low: **one rendered video file, plus real numbers.**

| Field | Why |
|---|---|
| GPU name and VRAM | Colab varies. Every number is meaningless without it |
| `peak_vram_mib` | Must be in the **thousands**. 3 means the GPU was never touched |
| Weights on disk | Must be **several GB**. 96MB means the download failed again |
| Cold and warm inference seconds | Report both; the first call is always slower |
| Output resolution and frame count | fps means nothing without resolution |
| **Render time ÷ audio duration** | Below 1.0 is faster than real time. The viability number |
| Identity-prep seconds | Offline and one-time. Slow here is fine and architecturally significant |

The notebook refuses to print an fps number when no output file exists. That guard is why
no fabricated figure entered the write-up after run 1, and it stays.

## What this unblocks

M0 → **M2**, which is my work and not large: `renderers/musetalk.py` implementing the
existing `TalkingHeadRenderer` Protocol. Face detection, parsing, and latent encoding of
the reference frames go in `prepare_identity`, which is explicitly allowed to be slow
because it runs once. The state machine, transport, client, and 199 tests do not change —
that is what the boundary was built for.

Worth keeping in proportion, though: `PROCESS.md` §3.3.3 measured a full turn at
**3.7–5.4s**, and **none of the three dominant terms is the renderer**. M0 gets you a face.
It does not get you a fast conversation.
