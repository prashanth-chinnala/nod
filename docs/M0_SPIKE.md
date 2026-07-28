# M0 — model spike runbook

**Timebox: one day.** Not one day of effort — one calendar day, after which you stop and
escalate regardless of how close it feels.

M0 answers exactly one question:

> Does a candidate talking-head model run at all on the hardware I actually have, and how
> fast?

Everything downstream of the answer is blocked on it: M2 (real renderer), the model-selection
memo (§2 of `PROCESS.md`), the headline latency numbers (§3.3), and the cost side of
build-vs-buy (§4.2). Nothing else in the assessment is blocked, which is why M1 and M3 were
built first.

The spike is **throwaway**. It happens in a scratch directory or a Colab runtime, *not* in
this repo. What comes back into the repo is two things: a set of numbers, and a log of every
setup problem you hit. The second one is worth more than it sounds — it is the evidence that
turns "I picked MuseTalk" into a defensible memo, and §2.1 of `PROCESS.md` has a
"setup fragility" criterion with nothing in it.

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
| Output quality against a vendor | Fidelity comparisons. §4 of the brief puts that out of scope anyway |

If a number would be useful and you did not measure it, write `NOT YET MEASURED` and move on.
A plausible invented figure is the single worst outcome in this assessment.

---

## 2. Which model, and why this order

Your hardware is **Colab / Kaggle free tier — T4, 16GB**. That fact does most of the choosing.

**Try MuseTalk first.** (`github.com/TMElyralab/MuseTalk`)

- MIT-licensed code, weights permit commercial use
- Ships a documented real-time inference path with the identity preprocessing split out from
  per-frame inference, which is the split §1.2 of the architecture document turns on
- Plain PyTorch — no build step that has to match your exact GPU

**Try Ditto second, and only if MuseTalk fails.** (Ant Group, Apache-2.0)

Architecturally it is the better fit for this brief: designed for streaming, low first-frame
delay, ships a streaming pipeline. But it wants TensorRT 8.6.1 with **GPU-specific prebuilt
engines**, and an ephemeral Colab runtime is close to the worst possible host for that — the
engine you build in one session may not match the GPU you get in the next. That fights the
clean-clone requirement in §6 of the brief directly.

That trade-off — better architecture, worse operability on the hardware available — is
exactly the kind of reasoning §2.3 is asking for. Whichever way you land, write down the
argument *against* your pick.

**Do not spend time on:**

- **Wav2Lip** — the licence prohibits commercial use. Useful as a documented rejection on
  licence, which §2.2 explicitly asks for.
- **LatentSync** — roughly 10× slower than real time. Useful as a documented rejection on
  latency, which §2.2 also asks for.

You get both required rejections without running either. Cite the licence and the published
throughput.

---

## 3. Running it

Open [`notebooks/m0_musetalk_spike.ipynb`](../notebooks/m0_musetalk_spike.ipynb) in Colab
(`File → Upload notebook`), set the runtime to a **T4 GPU**, and run the cells top to bottom.

It is deliberately not a one-click script. Each cell is timed and prints what it did, so when
something breaks you know which step broke and how long you had spent — which is the log
§2.1 needs.

The final cell prints a JSON block. Paste that back and it goes into `PROCESS.md`.

**Two things the notebook cannot do for you:**

1. **Check the upstream README before running the inference cells.** MuseTalk's invocation
   has changed across versions (v1.0 → v1.5 added a `--version` flag, and the config format
   moved). The notebook uses the current documented form, but upstream is the authority, not
   this repo. If a cell fails on an unknown argument, that is what happened.
2. **Decide when to stop.** See §5.

---

## 4. What to record

### 4.1 The numbers

The notebook captures all of these. If you end up running things by hand, these are the ones
that matter:

| Metric | Why it matters | Where it lands |
|---|---|---|
| GPU model and VRAM, actual | Colab hands out T4 / L4 / A100 unpredictably. Every number below is meaningless without this | §3.3 hardware column |
| Identity prep, wall clock | Offline, one-time, per-persona. If it is slow that is *fine* and architecturally important | §1.2 enrollment latency |
| Inference fps, cold run | First call is usually far slower — CUDA context, cuDNN autotune, lazy weight loads | §2.2, §3.3 |
| Inference fps, warm run | The number that describes steady state. Report **both** | §2.2, §3.3 |
| Output resolution | fps means nothing without it | §3.3 |
| Peak VRAM | Decides how many sessions fit one GPU, which drives §4.2's capacity line | §3.3 |
| Ratio of render time to audio duration | Below 1.0 is faster than real time. This is the number that says whether it is viable at all | §3.4 |

### 4.2 The setup log — do not skip this

Keep a running list as you go. Every entry is one line:

```
14:05  torch version in the Colab image conflicts with requirements.txt pin — pip resolved it, 4 min
14:20  download_weights.sh failed on the dwpose checkpoint, 403 — retried, worked second time
14:35  inference.py rejected --version, this build predates v1.5 — dropped the flag
```

This becomes `PROCESS.md` §2.1's "setup fragility" row and §2.2's verdict column. Reconstructing
it from memory on day 12 produces a much weaker memo than writing it as you go — which is the
same reason `DEVLOG.md` exists.

---

## 5. When to stop

Stop and escalate — meaning: tell me, and we replan — if any of these happen:

- **Four hours in and no rendered frame yet.** Not "nearly working." No output file.
- **The weights will not download.** Some MuseTalk mirrors have been flaky. Try Hugging Face
  directly before concluding.
- **Colab keeps giving you a runtime without a GPU,** or disconnects before a run finishes.
- **It runs but at under ~5fps at the smallest resolution.** That is not a tuning problem,
  that is the wrong model for this hardware.

Do not spend three days fighting a CUDA install. The assessment does not reward it, and the
options if MuseTalk fails are all cheap: try Ditto, drop the resolution, or accept a
CPU-only lighter model and report real numbers — which §5 of the brief explicitly permits as
long as you say so in the memo.

---

## 6. What happens when you come back

Paste me the JSON block and the setup log. Then:

1. The numbers go into `PROCESS.md` §2.2 and §3.3, attributed to the actual GPU you got.
2. The setup log becomes §2.1's fragility row and §2.2's verdict column.
3. **You** write §2.3 — the pick, the decisive criterion, the strongest argument against it,
   and what would make you switch. That one is `[HUMAN]`; I will not draft it.
4. M2 unblocks: I implement the renderer behind the existing `TalkingHeadRenderer` Protocol.
   Model-specific preprocessing goes in `prepare_identity`, which is allowed to be slow.
   The state machine does not change — that is the whole point of the boundary.

**Meanwhile:** M4 is not blocked on any of this. VAD, real STT, a real LLM adapter and real
TTS all sit behind Protocols that already exist and already have tests. If the GPU is going
to take a few days, that work can happen in parallel.
