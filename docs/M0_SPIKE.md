# M0 — model spike runbook

**Timebox: one day.** Not one day of effort — one calendar day, after which you stop and
escalate regardless of how close it feels.

M0 answers exactly one question:

> Does a candidate talking-head model run at all on the hardware I actually have, and how
> fast?

Everything downstream of the answer is blocked on it: M2 (real renderer), the
model-selection memo (§2 of `PROCESS.md`), the model rows of the latency numbers (§3.3),
and the cost side of build-vs-buy (§4.2). Nothing else in the assessment is blocked, which
is why the session layer was built first.

The spike is **throwaway**. It happens in a Colab runtime, *not* in this repo. What comes
back is two things: a set of numbers, and a log of every setup problem hit. The second is
worth more than it sounds — it is what turns "I picked MuseTalk" into a defensible memo,
and §2.1 of `PROCESS.md` has a "setup fragility" criterion that only a real log can fill.

**Status: run 1 failed in setup. Run 2 has not been attempted.** §4 is the post-mortem.

---

## 1. Before you start: what you are allowed to conclude

Read this first, because the failure mode here is answering a different question than the
one asked.

| You measure | You may conclude |
|---|---|
| 14fps at 256×256 on a T4 | This model, at this resolution, on this GPU, is below real time |
| Prep takes 90s, inference 2s | Identity preprocessing is offline work — architecturally significant, see §1.2 |
| Peak VRAM 6.2GB | It fits a T4's 16GB with room for a warm pool of 2 |
| Two hours lost to a CUDA/TensorRT mismatch | Setup fragility is real for this model, and belongs in the memo |

| You did **not** measure | So do not write |
|---|---|
| Anything about an L4, A10G, or A100 | "An L4 would give us 30fps" — unless you rent one and run it |
| End-to-end conversational latency | Anything in §1.5's measured column. That needs M2 wired into the prototype |
| Output quality against a vendor | Fidelity comparisons. The brief puts that out of scope anyway |

If a number would be useful and you did not measure it, write `NOT YET MEASURED` and move
on. A plausible invented figure is the single worst outcome in this assessment.

---

## 2. Which model, and why this order

The hardware is **Colab / Kaggle free tier — T4, 16GB**. That fact does most of the
choosing.

**Try MuseTalk first.** (`github.com/TMElyralab/MuseTalk`)

- MIT-licensed code, weights permit commercial use
- Ships a documented real-time inference path with identity preprocessing split out from
  per-frame inference, which is the split §1.2 of the architecture document turns on
- Plain PyTorch — no build step that has to match your exact GPU

**Try Ditto second, and only if MuseTalk fails.** (Ant Group, Apache-2.0)

Architecturally it is the better fit for this brief: designed for streaming, low
first-frame delay, ships a streaming pipeline. But it wants TensorRT 8.6.1 with
**GPU-specific prebuilt engines**, and an ephemeral Colab runtime is close to the worst
possible host for that — the engine built in one session may not match the GPU handed out
in the next. That fights the clean-clone requirement in §6 of the brief directly.

That trade-off — better architecture, worse operability on the hardware available — is
exactly the reasoning §2.3 is asking for. Whichever way it lands, write down the argument
*against* the pick.

**Do not spend time on:**

- **Wav2Lip** — the licence prohibits commercial use. Useful as a documented rejection on
  licence, which §2.2 explicitly asks for.
- **LatentSync** — roughly 10× slower than real time. Useful as a documented rejection on
  latency, which §2.2 also asks for.

Both required rejections come free, from the licence and the published throughput.

---

## 3. Running it

One command, in any GPU environment — Colab in a browser, Colab via the VS Code
extension, Kaggle, or a rented box:

```
!git clone -q https://github.com/prashanth-chinnala/nod.git \
  && python nod/scripts/m0_spike.py
```

`scripts/m0_spike.py` prints a JSON block. Paste it back. **Exit codes are meaningful:**

| Code | Meaning |
|---|---|
| `0` | Real numbers |
| `1` | Inference did not actually run |
| `2` | No GPU |
| `3` | Python version mismatch — take Route A below |
| `5` | Imports unresolved |
| `6` | Bad checkpoints |

It refuses to skip three gates, each corresponding to a way run 1 produced
plausible-looking numbers that measured nothing:

- **Imports must resolve before the download.** Run 1 fetched, then discovered nothing
  imported.
- **Every checkpoint is audited** — by first bytes, not just presence, so a git-lfs
  pointer file or a Google Drive quota HTML page is caught. Those produce a file of the
  wrong *kind*, which is what makes them silent.
- **`inference_actually_ran`** is computed from exit code **and** an output file **and**
  VRAM passing 500 MiB. Run 1 reported a 3 MiB peak beside a 15.43-second "warm
  inference" — the time was real and measured how long the process took to fail.

Use [`notebooks/m0_musetalk_v2.ipynb`](../notebooks/m0_musetalk_v2.ipynb) instead to step
through it, or for Route A's conda install.

### Route A — pin Python 3.10 with condacolab (recommended)

Fixes cause 3 below head-on and stays on MuseTalk's documented, tested stack. Costs one
runtime restart and ~10 minutes.

1. `pip install -q condacolab` then `condacolab.install()` — **the runtime restarts**,
   which is expected, not a crash.
2. Create a 3.10 environment; install the pinned torch and the OpenMMLab stack via
   `openmim`.
3. Download weights with `HF_ENDPOINT` **unset**, so it talks to Hugging Face directly.
4. **Verify every checkpoint's size before running anything.** This is the step whose
   absence cost run 1.
5. Run the v1.5 command with all four required arguments (see cause 1).

### Route B — accept the version drift

Stay on Python 3.12, install the newest `mmcv`/`mmpose` that have 3.12 wheels, and see
whether MuseTalk's code still runs against them. Faster to attempt, and it may fail on an
API change inside `mmpose` that you would then be debugging instead of measuring.

**Worth 30 minutes, not worth 3 hours.** If it fails, take Route A.

---

## 4. Run 1 — what actually went wrong

Three causes, established by reading MuseTalk's own `download_weights.sh` and README
rather than by guessing. **One of them was a bug in this repo's notebook.**

### Cause 1 — the inference command was incomplete (our error)

The first notebook ran:

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
`exit_code: 1`.

### Cause 2 — the weights download fails silently, by design

`download_weights.sh` sets **`HF_ENDPOINT` to a mirror** rather than Hugging Face itself,
then runs a sequence of `huggingface-cli download --include ...` calls, plus `gdown` for
one file from Google Drive and `curl` for the ResNet18 backbone.

Three ways that produces 96 MB and exit code 0:

- The mirror is frequently unreachable or rate-limited from Colab. Each failed
  `huggingface-cli` call prints an error and returns, but the script has no `set -e`, so
  the shell exits 0 regardless.
- `--include` filters can match nothing and still succeed.
- `gdown` on a Google Drive link hits the interstitial quota page and writes an HTML file
  where a checkpoint should be.

**Nothing validates a single byte.** 96 MB is roughly `resnet18` (45 MB) plus small
configs — i.e. only the `curl` step worked.

### Cause 3 — Python 3.12 versus a Python 3.10 pinned stack

The runtime reported **`python: 3.12.13`**. MuseTalk pins:

| | Pinned |
|---|---|
| Python | 3.10 |
| torch | 2.0.1 (cu118) |
| mmcv | 2.0.1 |
| mmdet | 3.1.0 |
| mmpose | 1.1.0 |

There are no prebuilt `mmcv==2.0.1` wheels for Python 3.12, and building from source needs
a matching CUDA toolchain. This explains the other implausible number: **`pip install`
finished in 13 seconds** for a stack that normally takes minutes. It did not install the
OpenMMLab packages — it resolved what it could and moved on.

**That mismatch is itself a §2.1 finding.** A model whose documented stack no longer
installs on the current free-tier runtime has a real clean-clone problem, and that belongs
in the memo whichever model is picked.

---

## 5. What run 2 must produce

The bar is unchanged and low: **one rendered video file, plus real numbers.**

| Field | Why |
|---|---|
| GPU name and VRAM, actual | Colab hands out T4 / L4 / A100 unpredictably. Every number below is meaningless without it |
| `peak_vram_mib` | Must be in the **thousands**. 3 means the GPU was never touched |
| Weights on disk | Must be **several GB**. 96 MB means the download failed again |
| Identity prep, wall clock | Offline, one-time, per-persona. Slow here is *fine* and architecturally significant |
| Inference fps, cold **and** warm | The first call is always slower — CUDA context, cuDNN autotune, lazy weight loads. Report both |
| Output resolution and frame count | fps means nothing without resolution |
| **Render time ÷ audio duration** | Below 1.0 is faster than real time. The viability number |

The harness refuses to print an fps figure when no output file exists. That guard is why
no fabricated number entered the write-up after run 1, and it stays.

### The setup log — do not skip this

Keep a running list. Every entry is one line:

```
14:05  torch version in the Colab image conflicts with the requirements pin — pip resolved it, 4 min
14:20  download_weights.sh failed on the dwpose checkpoint, 403 — retried, worked second time
14:35  inference.py rejected --version, this build predates v1.5 — dropped the flag
```

This becomes `PROCESS.md` §2.1's fragility row and §2.2's verdict column. Reconstructing
it from memory later produces a much weaker memo than writing it as you go — the same
reason `DEVLOG.md` exists.

---

## 6. When to stop

Stop and escalate — meaning: replan — if any of these happen:

- **Four hours in and no rendered frame.** Not "nearly working." No output file.
- **The weights will not download.** Try Hugging Face directly, with `HF_ENDPOINT` unset,
  before concluding.
- **Colab keeps handing out a runtime without a GPU,** or disconnects before a run
  finishes.
- **It runs but under ~5fps at the smallest resolution.** That is not a tuning problem,
  that is the wrong model for this hardware.

Do not spend three days fighting a CUDA install. The assessment does not reward it, and
the alternatives are all cheap:

- **Try Ditto.** Apache-2.0, streaming-native. Its TensorRT requirement fights an
  ephemeral runtime, which is a documented cost rather than a surprise.
- **Drop the resolution** and report the real number at that resolution.
- **Report the dead end.** "MuseTalk's documented stack does not install on the current
  free Colab runtime, and its weight downloader exits 0 having fetched 2% of the
  checkpoints" is a *finding*, not a failure to produce one. §2.1's setup-fragility
  criterion is asking for precisely this, and §5 of the brief explicitly permits a lighter
  model with real numbers as long as the memo says so.

---

## 7. What happens afterwards

Paste back the JSON block and the setup log. Then:

1. The numbers go into `PROCESS.md` §2.2 and §3.3, attributed to the GPU actually handed
   out.
2. The setup log becomes §2.1's fragility row and §2.2's verdict column.
3. **You** write §2.3 — the pick, the decisive criterion, the strongest argument against
   it, and what would make you switch. That one is `[HUMAN]`.
4. M2 unblocks: the renderer goes in behind the existing `TalkingHeadRenderer` Protocol.
   Model-specific preprocessing goes in `prepare_identity`, which is allowed to be slow
   because it runs once. **The state machine, transport, client, and tests do not change**
   — that is the whole point of the boundary, and it is worth demonstrating live on the
   Loom rather than asserting.

**Keep it in proportion.** `PROCESS.md` §3.3.3 measured a full turn at **3.7–5.4s**, and
**none of the three dominant terms is the renderer**. A perfect zero-latency model still
leaves ~3.4s. M0 gets a face. It does not get a fast conversation.
