# PROCESS.md — engineering log

> **This is a historical document, kept deliberately.** It records the first phase of the project:
> the architecture research, the model-selection reasoning, and the measured latency budget, written
> when no talking-head model was integrated and everything ran on an M1 Pro with no CUDA.
>
> **It is out of date as a description of the system.** A real face renders on a GPU now, the store
> is Postgres, and several figures here were superseded — including two that turned out to be wrong
> because they were measured on the wrong device. For the current state read
> [ARCHITECTURE.md](ARCHITECTURE.md), [MODELS.md](MODELS.md), [MEASUREMENTS.md](MEASUREMENTS.md) and
> [OPERATIONS.md](OPERATIONS.md).
>
> It survives because the reasoning is still the reasoning: why the module boundary exists, why
> cancellation is an integer, why turn detection stayed local, and what the alternatives were. A
> decision log is worth more when it includes the state of knowledge at the time — including the
> parts later proved wrong. Corrections are recorded in MEASUREMENTS.md rather than edited into the
> text here.

Real-time conversational avatar — architecture research, prototype, and model-selection reasoning.

| | |
|---|---|
| Author | Prashanth Chinnala |
| Repository | <https://github.com/prashanth-chinnala/nod> |
| Time spent | ~1.5 days. The brief suggests 10–14; this was written against a one-day-old deadline, and §3.1 records what that traded away |
| Hardware used *(when this was written)* | **Apple M1 Pro, 16GB, no CUDA** — everything measured **in this document** ran on this. A free-tier **Colab T4 16GB** was used to attempt the model spike; it failed in setup before reaching the GPU (§2.2.1) |
| Prototype status *(when this was written)* | A real spoken conversation worked end to end — real transcription, a real LLM, a real synthesised voice, session lifecycle, and interruption. **No talking-head model was integrated**; see §3.1 and §3.4 |
| **Prototype status today** | **A real face renders on a Tesla T4**, from an uploaded video or from a photograph animated at enrollment. Measured: **78.4 ms/frame render, 8.3 fps delivered** against an 8 fps target, trailing audio/video gap **−66 ms to +172 ms**, first turn no slower than the fifth. Not real time — 2.0× short of 25 fps, and VAE decode is 74% of a frame. Voice cloning works but not on the same card as the face. Figures and method: [MEASUREMENTS.md](MEASUREMENTS.md) §2 and §8b |
| **Where §3.3 and §3.4 were superseded** | Every `NOT YET MEASURED` row below has an answer now, and two figures here were later shown wrong because they were measured on the wrong device (batch size, fp16 ratio). The corrections live in [MEASUREMENTS.md](MEASUREMENTS.md), not edited into the text below — see the banner |

## Contents

1. [Architecture document](#1-architecture-document) — how a Tavus-class system works
2. [Model selection memo](#2-model-selection-memo) — what I evaluated and picked
3. [Prototype scope and process](#3-prototype-scope-and-process) — what I built, deferred, and measured
4. [Build-vs-buy memo](#4-build-vs-buy-memo) — the recommendation
5. [Migration plan](#5-migration-plan) — vendor cutover without regression
6. [Sources](#6-sources)

---

## 0. Claim tagging convention

Every substantive claim in §1 carries one of:

| Tag | Meaning |
|---|---|
| **[C]** confirmed | A primary source states this. Citation in §6. |
| **[I]** inferred | My engineering judgment from observable behavior, adjacent published work, or physical constraint. No source states it. |
| **[U]** unknown | I could not determine this, and it matters. Stated rather than papered over. |

---

## 1. Architecture document

### 1.1 System overview

A production Tavus-class system is not one model. It is a **pipeline of streaming stages
wrapped in deterministic session orchestration**, where the talking-head model is one
bounded stage near the end. The interesting engineering is almost entirely in the
orchestration: which stages overlap, who is allowed to cancel whom, and what happens to
work already in flight when the candidate interrupts.

The single most important structural claim in this document: **the stages must overlap.**
Executed sequentially and to completion, the same components produce a multi-second
turnaround with every one of them performing exactly to spec. Executed as overlapping
streams — LLM tokens chunked into sentences as they arrive, each sentence sent to TTS
before the next is generated, each audio chunk driving frames before the sentence finishes
— the first frame can emerge while the LLM is still writing. That is the whole trick, and
§3.3.3 shows what it costs when a link in the chain is slow anyway.

```
        ┌────── candidate speaks ───────┐
        │                               │
        ▼                               │
   ┌─────────┐   audio frames (32ms)    │
   │  MIC    ├──────────────┬───────────┘
   └─────────┘              │
                            ▼
                  ┌───────────────────┐
                  │ TURN DETECTION    │  ◀── streaming, local. NOT the vendor's
                  │ VAD + hysteresis  │      endpointing: this decides the turn
                  └─────────┬─────────┘
                            │  end-of-turn (silence window elapsed)   ▲ BLOCKING
                            ▼                                        │ the wait IS
                  ┌───────────────────┐                              │ the policy
                  │ STT               │  ◀── streaming; mostly ──────┘
                  │                   │      already done by now
                  └─────────┬─────────┘
                            │  final transcript
                            ▼
                  ┌───────────────────┐
                  │ LLM               │  ◀── streaming. Only TTFT matters.
                  └─────────┬─────────┘      Total generation time does not.
                            │  tokens ──▶ sentence chunker
                            ▼
                  ┌───────────────────┐
                  │ TTS               │  ◀── streaming, per sentence.
                  └─────────┬─────────┘      Sentence 2 synthesises while
                            │  PCM chunks     sentence 1 is still playing.
                            ▼
                  ┌───────────────────┐
                  │ AVATAR RENDER     │  ◀── streaming. Needs only the first
                  │ identity + audio  │      ~200ms of audio to emit frame 1.
                  └─────────┬─────────┘
                            │  frames + audio, tagged with a turn epoch
                            ▼
                  ┌───────────────────┐
                  │ MIXER             │  ◀── constant cadence. Owns presentation
                  │ idle loop ⇄ turn  │      timestamps. Never stalls the track.
                  └─────────┬─────────┘
                            ▼
                  ┌───────────────────┐
                  │ TRANSPORT         │  ── WebRTC in production;
                  │                   │     WebSocket in this prototype (§1.4)
                  └─────────┬─────────┘
                            ▼
                       ┌────────┐
                       │BROWSER │ ──▶ reports first paint + audio actually played
                       └────────┘
```

**Streaming vs. blocking, which is the point of the diagram:** every stage above is
streaming *except* end-of-turn detection. That stage is a deliberate wait — and in this
prototype it is **700ms**, the single largest term in the measured budget (§1.5). It is
not a performance bug to be optimised away; it is a conversational policy choice, and no
amount of hardware changes it. Cutting it to 300ms makes the system interrupt people
mid-thought.

### 1.2 Identity capture

The load-bearing question is **per-person weights vs. a shared model plus per-person
precomputed features**, because the two have completely different serving costs. Per-person
weights mean cold-loading a checkpoint onto a GPU worker when a session starts. A shared
model with cached identity features means the weights are already resident and only a small
per-person artifact needs fetching.

MuseTalk — the open-source system this prototype targets, and the one where the internals
are readable rather than inferred — is firmly the second kind. Its weights are
identity-agnostic; a reference video becomes a set of precomputed *features*, not a
fine-tune.

| Question | Answer | Evidence | Tag |
|---|---|---|---|
| Input to enrollment | A short reference video or a single image of a face, roughly front-on. MuseTalk operates on a **256×256 face region** cropped from the frame, so framing matters more than duration | MuseTalk README, read directly | **[C]** for MuseTalk. **[I]** that vendors accept similar input |
| Artifact produced | Per-frame VAE latents of the reference frames, plus face detection / parsing / bounding-box metadata. **No per-person model weights** | MuseTalk README + inference code | **[C]** for MuseTalk. **[I]** for vendors — a per-person-weights design is possible and would change their cost structure |
| Enrollment latency | Dominated by face detection and VAE encoding over the reference frames — seconds to minutes depending on clip length. `NOT YET MEASURED` here (blocked on the model spike) | Not measured | **[U]** — I did not run it, and vendors do not publish enrollment timings |
| Reusable across sessions? | Yes. The artifact is a function of the reference clip alone, so it is computed once per persona and cached | Follows from the above | **[I]** — a deduction from the artifact being identity-agnostic, not a documented statement |
| Per-person GPU state at inference | Only the cached latents and crop metadata. The U-Net, VAE, and audio encoder are shared across every persona on the worker | MuseTalk README | **[C]** for MuseTalk. **[I]** for vendors |

**The serving consequence, stated plainly:** because identity is data rather than weights,
one warm GPU worker can serve any persona without reloading a model. That is what makes a
warm pool (§1.4) economically viable at all. If the architecture required per-person
weights, every session start would pay a checkpoint load, and the pool would have to be
partitioned by persona rather than shared — a materially worse cost curve.

**Why this is architecturally significant beyond cost:** the existence of a precomputation
step is *the reason first-frame latency can be low at conversation time.* All the expensive
identity work happens offline. At conversation time the model is doing something much
smaller — conditioning cached latents on new audio. This is exactly why the
`TalkingHeadRenderer` Protocol in this prototype splits `prepare_identity` from
`push_audio`/`frames`: the boundary mirrors the architectural split, and `prepare_identity`
is explicitly allowed to be slow.

### 1.3 The audio-to-video mechanism

Structured as **elimination**, because the real-time constraint kills most of the design
space and the reasoning is worth more than the answer. The bar: ~25–30fps sustained, and a
first frame inside a sub-second turn budget that STT, LLM, and TTS have already spent most
of.

| Class | Example work | Why it survives or dies | Tag |
|---|---|---|---|
| Full generative video diffusion | Sora-class, Stable Video Diffusion | **Dies on throughput by orders of magnitude.** Generating a whole talking human per frame solves a vastly larger problem than needed, and nothing in this class runs at 25fps. Also uncontrollable: no guarantee the identity stays stable across a 20-minute interview | **[I]** |
| Audio-conditioned latent diffusion, multi-step | Diffusion talking-head research generally | **Dies on the step count.** *N* denoising steps per frame multiplies per-frame cost by *N*. At 25fps there is a ~40ms budget per frame; multi-step diffusion is nowhere near it | **[I]** — the 40ms budget is arithmetic; the conclusion is judgment |
| Motion-space diffusion + neural render | **Ditto** (Ant Group, ACM MM 2025) | **Survives, with an operational cost.** Diffuses in a compact *motion* space rather than pixel space, cuts denoising from **50 steps to 10**, and compiles the DiT to **TensorRT**. Explicitly built for streaming and low first-frame delay. The TensorRT engines are GPU-specific, which is a real deployment constraint | **[C]** — the 50→10 steps and TensorRT are stated in the paper and repo |
| Latent-space mouth inpainting, single-step | **MuseTalk** | **Survives, and is the cheapest.** Borrows the Stable Diffusion v1.4 U-Net architecture but **is not a diffusion model** — it inpaints the mouth region in VAE latent space in a **single step**. Claims **30fps+ on a V100** at a 256×256 face region | **[C]** — the repo states all three, including "NOT a diffusion model". **Unverified by me**: the spike never ran |
| 3D rig driven by visemes | Classical game-engine avatars | **Survives on latency, dies on realism.** Viseme-driven rigs are trivially real-time and fully controllable, but look animated rather than photographic — the wrong product for an interview meant to feel human | **[I]** |
| 3D Gaussian splatting / NeRF per-identity | Per-identity neural head research | **Dies on enrollment, not inference.** Inference can be fast, but each identity needs its own trained representation — minutes to hours of per-person GPU work. That breaks the "upload a reference clip, interview in seconds" product shape and reintroduces the per-person-weights cost §1.2 rejected | **[I]** |

**What survives** is a narrow band: *single- or few-step generation, in a compact latent or
motion space, over a small region of the frame, conditioned on precomputed identity
features.* That is a much smaller problem than "generate video of a person talking," and
recognising that it is smaller is the actual insight.

#### 1.3.1 Two sub-claims that separate a surface answer from a real one

**Sub-claim A — most of the frame is replayed, not generated. [C] for MuseTalk, [I] for vendors.**

The strong inference is that production systems synthesise only the **mouth / lower-face
region** and composite it onto pre-recorded body footage, rather than generating a whole
human per frame. MuseTalk does exactly this and is explicit about it: it inpaints a 256×256
face region and leaves the rest of the frame alone.

How to verify it from observed vendor output, without any inside access:

- **A fixed or near-fixed background** across a long session
- **Repeating gesture and blink cycles** — a tell that body motion is a looped clip
- **Seam artifacts at the jawline or neck** under fast speech, where the composite boundary
  sits
- **Head pose that does not respond to speech content** — the head moves on the loop's
  schedule, not the sentence's

**Sub-claim B — the audio conditioning signal is self-supervised speech encoder features,
not raw waveform or bare mel spectrograms. [C] for MuseTalk, [I] for vendors.**

This has a strong primary-source anchor in open source: MuseTalk encodes audio with a
frozen **`whisper-tiny`** model and fuses those embeddings into the U-Net's image
embeddings by **cross-attention**. Whisper-encoder / HuBERT / wav2vec2 features are the
norm across published real-time systems.

The engineering reason: these features are already phonetically structured and
speaker-normalised, so the visual model learns a much easier mapping than it would from
spectrograms — and it generalises across voices it never trained on, which is what makes
one model serve any TTS voice.

#### 1.3.2 Temporal consistency across chunk boundaries

This is where naive streaming implementations visibly break, and it is worth being concrete
because the failure is characteristic: a **flicker or jaw jump exactly at chunk
boundaries**, periodic at the chunk rate. Once seen it is unmistakable, and it is the first
thing to look for in any streaming talking-head demo.

Three mechanisms address it, and they compose:

1. **Overlapping audio context.** Condition each chunk on a window that extends before and
   after its own frames, so a frame's generation sees the phonemes on both sides of it.
   MuseTalk's audio feature extraction takes a multi-frame window rather than a single
   frame's worth.
2. **Anchoring to shared reference latents.** Because every frame is inpainted against
   *the same* cached identity latents (§1.2), there is no drift in appearance between
   chunks — the identity cannot wander, because it is not being generated.
3. **Compositing only the region that changes.** If the background and body come from a
   fixed clip, they are identical across a boundary by construction. Only the mouth region
   can flicker, which shrinks the problem to a small area.

**A fourth, at the orchestration layer rather than the model:** the mixer must emit at a
constant cadence and own presentation timestamps, so that a renderer briefly falling behind
produces a repeated frame rather than a stalled track. A stall is more visible than a
duplicate — and it also corrupts the receiver's jitter estimate, so the recovery is worse
than the original glitch. That is implemented and tested in this prototype
(`src/avatar/mixer.py`), and it is the one part of this section that is not inference.

### 1.4 Real-time serving architecture

> **This section is a scaffold — the italics below are prompts, not prose.** So are §1.6 and §1.7.
> Drafts for all three, with proposed tags and thresholds, are in
> [docs/DRAFT_FOR_REVIEW.md](docs/DRAFT_FOR_REVIEW.md) awaiting review; they are proposals and are
> deliberately not merged here, because the tags are the author's to assert. Flagged rather than left
> to be discovered: the brief lists this document as deliverable #1 and asks for the observability
> instrumentation by name.

**Session and model pooling.** *Cold-start cost of loading model weights + CUDA context, why that cannot be paid at conversation start, what a warm pool implies for cost (idle GPU time you pay for), how sessions bind to workers, and what happens under a thundering herd of simultaneous interviews.*

**Frame transport.** *WebRTC vs. WebSocket vs. HLS/DASH. Kill HLS on segment granularity. Justify WebRTC on jitter buffering, congestion control, NAT traversal, and browser-native decode. Note whether an SFU sits in the path and what that adds in latency and operational surface.*

**Audio/video sync.** *Whose clock wins, and how lip-sync survives network jitter. If audio and video travel as separate tracks, what keeps them aligned?*

**The pipelining insight.** *State plainly that the stages must overlap: LLM tokens stream to sentence-chunked TTS, TTS audio chunks stream to the renderer, the first video frame emits after the first ~200ms of audio exists rather than after full synthesis. Sequential-and-complete execution of the same stages produces a multi-second turnaround with identical component performance.*

### 1.5 Latency budget

Measured on an Apple M1 Pro, 16GB, no GPU, over **three consecutive end-to-end runs** with
every component real — Deepgram Nova transcribing, `gpt-oss:20b` via Ollama Cloud, Deepgram
Aura-2 speaking, the placeholder renderer. Three runs is a small sample and the spread is
reported rather than averaged away, because the spread is the finding: the LLM term varies
by **2.9×** run to run, and a mean would hide that.

| Stage | Target (ms) | Measured (ms) — 3 runs | Tag | Notes |
|---|---|---|---|---|
| End-of-turn detection | 100–300 | **700** (fixed) | | Configuration, not measurement. Over twice the top of my own target, and deliberately so — see §3.3.1 |
| Speech-to-text finalize | 50–150 | **~0 observed** | | Streaming over a persistent socket, so the transcript is finalised by the time the silence window elapses. This term is hidden *inside* the 700ms above rather than added to it |
| LLM time-to-first-token | 200–500 | **1,645 / 2,942 / 4,724** | | `gpt-oss:20b` on Ollama Cloud's free tier. **3–9× over target and the least predictable term in the budget.** A paid low-latency endpoint is the fix; this is not a model-architecture problem |
| TTS time-to-first-audio | 100–300 | **869 / 956 / 889** | | Deepgram Aura-2 over REST. Remarkably stable, and ~3× over target. Aura's WebSocket interface measured **351–361ms** — verified, not built (§3.4) |
| Avatar first frame | 50–150 | **2,656 / 4,136 / 5,782** | | Placeholder renderer, so this is *not* a model figure: it is the LLM plus TTS plus a few ms. A real model adds to it |
| Encode + network + jitter buffer | 50–150 | **20–25** | | Loopback, so this is a floor, not a realistic figure. PNG encode plus socket plus decode plus paint. Across a real network the client's 150ms jitter buffer is added on top |
| **Perceived total** | **< 1,000** | **2,679 / 4,161 / 5,802** | | Turn start to this page finishing `drawImage`, reported by the browser. **3–6× the target** |

**Which term I would attack first, and why it is not the obvious one.** The LLM is the
largest and worst-behaved term, but the *cheapest* real win is TTS: swapping Aura REST for
its WebSocket interface is measured at ~550ms for about a day's work, with no quality
trade-off. The LLM term is bigger but the fix is commercial rather than engineering — pay
for a low-latency endpoint. The 700ms end-of-turn window is the second largest single term
and **no amount of hardware touches it**; it is a conversational judgment, and the only real
attack on it is speculative execution, which trades wasted compute for latency.

**The conclusion this table exists to support:** a full turn is **2.7–5.8s** against a
sub-second target, and **none of the three dominant terms is the renderer.** Subtract the
renderer entirely — set it to zero — and roughly 2.6–5.7s remains. "We need more GPU" is
measurably the wrong diagnosis for this pipeline.

### 1.6 Failure and edge handling

**Interruption (barge-in).**

**Silence.** *Idle-loop fallback. How a looping segment avoids a visible jump cut. What triggers a re-prompt, and after how long.*

**Reconnect.** *Where session state lives so a reconnect resumes rather than restarts. ICE restart vs. new session. How long a GPU worker is held before release, and the cost trade-off in that number.*

**Degradation ladder.** *What the system sheds first under GPU pressure — resolution, then fps, then falls back to audio-only with a static frame? A named ladder beats "it would degrade gracefully."*

### 1.7 Observability plan

| Signal | Type | Where emitted | Alert threshold |
|---|---|---|---|
| Per-stage latency | histogram, p50/p95/p99 | | |
| End-to-end turn latency | histogram | | |
| Interruption→silence latency | histogram | | |
| Frames dropped / late | counter | | |
| Lip-sync drift | gauge | | |
| GPU pool utilization & queue depth | gauge | | |
| Session failure by cause | counter, labeled | | |
| Reconnect rate | counter | | |

---

## 2. Model selection memo

### 2.1 Criteria and weights

Weighted before running anything, so the criteria could not be reverse-engineered from
whichever model happened to work. The weights are the argument: two of them are hard gates
rather than scores, because no amount of quality compensates for a licence that forbids the
use case.

| Criterion | Weight | Why it matters here |
|---|---|---|
| **License, code and weights** | **Gate** | Not a score. A model that cannot be used commercially is not a candidate, whatever it scores elsewhere — and code and weight licences differ, often materially |
| **Streaming-native vs. batch** | **Gate** | Also not a score. A batch model cannot be made streaming inside this time-box; that is a research project, not an integration |
| Achievable fps on accessible hardware | 30% | Real-time is the point, and "accessible" means the free-tier T4 actually available — not a rented A100 |
| First-frame latency | 25% | Distinct from throughput, and more important here. A model can hit 30fps and still have a slow first frame, and in a conversation the first frame is what the candidate waits for |
| Setup fragility | 20% | Weighted this heavily *because* the brief requires a clean-clone build, and it turned out to be the criterion that actually decided the outcome. Evidence: MuseTalk's `download_weights.sh` and `pip install` **both exited 0 without installing the model** — see §2.2.1 |
| Maintenance health | 15% | Commits, issue response, releases. A dead project is a liability the moment a CUDA version moves |
| Output quality at target resolution | 10% | Deliberately lowest. The brief puts production-grade visual fidelity explicitly out of scope, so weighting it higher would be optimising for something not being assessed |

### 2.2 Candidates evaluated

| Model | fps / latency | License (code / weights) | Streaming? | Verdict |
|---|---|---|---|---|
| MuseTalk (`0a89dec`) | **NOT YET MEASURED** — spike run 1 failed in setup before touching the GPU | MIT code; weights permit commercial use | Yes, documented realtime mode | Undecided. See §2.2.1 |
| Ditto | NOT YET MEASURED — not attempted | Apache-2.0 | Yes, streaming-native | Undecided. TensorRT 8.6.1 with GPU-specific prebuilt engines fights an ephemeral Colab runtime |
| Wav2Lip | Published ~real-time on modest GPUs | **Licence prohibits commercial use** | No | **Rejected on licence.** Not run |
| LatentSync | ~10x slower than real time (~100s for 10s of video on a 4090, published) | Open | No | **Rejected on latency.** Not run |

#### 2.2.1 Spike run 1 — MuseTalk on a free-tier Colab T4

**Outcome: failed in setup. No inference occurred, so nothing about this model's
throughput has been measured.** The three causes are below; one of them was a defect in my
own harness rather than in the model, and the harness now gates against all three
(`scripts/m0_spike.py`).

Three fields establish that no inference ran, and they matter more than the timings
alongside them:

| Field | Value | Reading |
|---|---|---|
| `peak_vram_mib` | 3 | The GPU was never used. A T4 running this model sits in the thousands. |
| `weights_on_disk` | 96M | The checkpoints total several GB. Almost nothing downloaded. |
| `inference_cold/warm.exit_code` | 1 | Failed. Both realtime runs failed identically. |
| output video | none | The harness refused to record an fps number, correctly. |

**Numbers from this run that must not be quoted as measurements:**
`cold_warm_ratio: 2.1` is the ratio between two crashes; `identity_prep_s: 0.25` is the
difference between two identical failures; `inference_warm.seconds: 15.43` is how long
it took to fail.

**What the run does establish, and what it is evidence for:**

1. **A free-tier T4 (15360 MiB, driver 580.82.07) was available on first attempt.** The
   hardware question this spike existed to answer is resolved.
2. **Setup fails silently.** `download_weights.sh` exited **0** having fetched 96MB, and
   `pip install -r requirements.txt` exited **0** in **13 seconds** for a project that
   depends on `mmcv` and `mmpose`. Both reported success without doing their job. That
   is the §2.1 "setup fragility" row, and it is a stronger signal than a throughput
   figure would have been: a model whose installer cannot tell you it failed is a model
   whose clean-clone story needs testing before it is trusted.

Setup timings to the point of failure, which are real: clone 1.6s, install 13.0s,
weights 15.6s.

### 2.3 Selection and rationale

**The pick: MuseTalk — and it has not been made to run.** Both halves of that sentence are
the answer, and separating them is the honest version of this section.

**The decisive criterion was licence, and it eliminated the field before performance was
considered.** Wav2Lip is the best-known model in this category and its terms are
unambiguous: *"This repository can only be used for personal/research/non-commercial
purposes"*, and because the weights are trained on LRS2, *"any form of commercial use is
strictly prohibited."* For a candidate-interview product that is the end of the
conversation — no fps figure could rescue it. MuseTalk is MIT for the code and its weights
are *"available for any purpose, even commercially."* That is the difference between a
model you can ship and a model you can demo.

**Among the two that clear both gates**, MuseTalk over Ditto, on operability rather than
architecture:

| | MuseTalk | Ditto |
|---|---|---|
| Approach | Single-step latent inpainting, mouth region only | Motion-space diffusion, 10 denoising steps |
| Published throughput | 30fps+ on a V100 | Real-time, low first-frame delay by design |
| Runtime | Plain PyTorch | **TensorRT 8.6.1, GPU-specific prebuilt engines** |
| Licence | MIT + commercial weights | Apache-2.0 |

Ditto's TensorRT requirement is the deciding factor and it cuts against it. Engines are
compiled per GPU architecture, and an ephemeral Colab runtime hands out a different GPU
between sessions — so the engine built in one session may not load in the next. That
directly fights the clean-clone requirement. MuseTalk being plain PyTorch means no build
step that has to match the hardware.

#### The strongest argument against my own pick

**Ditto is the better architecture for this problem, and I did not choose it.** It is built
for streaming with explicitly low first-frame delay; MuseTalk's real-time path is documented
but is not the design centre. In a world where I controlled the hardware — a fixed GPU
class, engines compiled once in CI — I think Ditto is the right call, and the reason I
didn't pick it is a property of my *hardware access*, not of the model. That is a weaker
justification than a technical one and I would rather say so than dress it up.

**The second argument against is more damaging: MuseTalk did not run.** Its documented stack
pins Python 3.10, torch 2.0.1+cu118, and mmcv 2.0.1; the current free Colab runtime is
Python 3.12, for which no `mmcv==2.0.1` wheels exist. So I picked the model that scored best
on setup fragility, and then setup fragility is precisely what defeated it. There is an
uncomfortable reading of that: perhaps my weighting was right and my *assessment* of
MuseTalk against it was wrong — MIT licensing and plain PyTorch made it look operable, and a
pinned OpenMMLab stack that no longer installs is exactly the fragility the criterion was
meant to catch.

**What is unmeasured, stated plainly:** I have no fps number, no VRAM figure, and no
first-frame latency for any talking-head model. §3.3's model rows say `NOT YET MEASURED`
because that is what they are. The selection is therefore made on licence, published
figures, and architecture — not on anything I observed on my own hardware.

#### What would make me switch

| Trigger | Threshold | Switch to |
|---|---|---|
| MuseTalk under Python 3.10 (condacolab) still fails | One more timeboxed attempt | Ditto, accepting the TensorRT cost |
| Measured throughput below real time on a T4 | < 15fps at 256×256 | Ditto, or drop resolution and report it |
| Deployment target becomes a fixed GPU class | Any commitment to owned or reserved hardware | **Ditto** — the objection above evaporates the moment engines can be compiled once |
| A newer model clears both gates with a simpler runtime | Published real-time on a T4, commercial weights | Re-run this comparison. The field moves fast enough that a decision this old deserves re-testing |

### 2.4 Why this is swappable

The `TalkingHeadRenderer` Protocol in [`src/avatar/contracts.py`](src/avatar/contracts.py) is
the entire surface the ML model is allowed to present to the rest of the system. Two
implementations ship: [`renderers/stub.py`](src/avatar/renderers/stub.py) (no GPU, used by
CI) and the chosen model. Session lifecycle, turn epochs, cancellation, frame cadence, and
transport all live outside it — see §3.2.

---

## 3. Prototype scope and process

### 3.1 Scoped in vs. deferred

| Capability | Status | Rationale |
|---|---|---|
| Session start/stop lifecycle | **Built** | State machine complete; 131 tests, all GPU-free |
| Idle-loop fallback frame | **Built** | `IdleLoop` + `FrameMixer`. The clip is a synthetic pulse, not a face — a real one needs a preparation step over a real reference clip |
| Interruption handling | **Built** | Turn-epoch cancellation, verified end-to-end including the client-side audio flush |
| Browser streaming transport | **Built** | WebSocket, the shortcut the brief permits. Costs stated in `transport/websocket.py` |
| Browser client | **Built** | Canvas + Web Audio + mic, plain JS, no build step |
| Playback acknowledgement | **Built** | Client reports how much audio actually played, including partial buffers stopped by a barge-in |
| End-to-end latency to browser paint | **Built** | Client reports first paint; the server cannot measure this for itself |
| **Audio in → lip-synced video out** | **Not built** | Blocked on the model spike. Needs a GPU, and no figure is written here that was not measured |
| **A talking-head model of any kind** | **Not built** | Same. `StubRenderer` proves the interface, not the capability |
| Real STT | **Built** | Deepgram Nova over a persistent WebSocket. Transcribes continuously; the local turn policy decides when the turn ends, not the vendor's endpointing — see the `Transcriber` docstring |
| Real TTS | **Built** | Deepgram Aura-2. `container=none` is load-bearing: the default response carries a 44-byte RIFF header that would be played as PCM |
| Real LLM | **Built** | Two adapters (Anthropic, OpenAI) behind one `SentenceStream`. `OPENAI_BASE_URL` also reaches Ollama / LM Studio / vLLM, so a local model needs no new adapter |
| Configuration | **Built** | `.env.development` / `.env.local` / `.env` layered at import, real env vars winning; `GET /config` reports what each boundary resolved to and which files were read. Every default is a working no-credential one |
| Turn-taking policy | **Built** | Server-side. Onset, hysteresis, retraction, and end-of-turn as separately tuned decisions; 30 tests over probability sequences |
| **A real voice activity detector** | **Partly** | The policy is real and tested. The default detector under it is an energy gate that cannot tell speech from a door. `SileroVad` is written, wired, and **has never been executed** — no torch in the dev environment |
| Frame encoding | **Built** | PNG, stdlib `zlib`, no new dependency. **108.10 KB → 0.57 KB per frame, 22.20 → 0.12 Mbps at 25fps — 188×.** Was the reason 0.5fps of 25 arrived through a tunnel. The client sniffs the format from magic bytes, so the real renderer can switch to JPEG for photographic frames with no protocol change |
| Client jitter buffer | **Built** | 150ms lead, `?audioLead=` overridable, underruns counted and surfaced. Absent before, which is why audio was clean on localhost and broke through a tunnel |
| Warm model pooling | Deferred | Described in §1.4. Constructing a renderer per session is exactly the cost that section argues cannot be paid at conversation start |
| Multi-session concurrency | Deferred | One orchestrator per socket is wired and works; only one session has been exercised |
| WebRTC transport | Deferred | Stretch goal. §1.4 states what the WebSocket shortcut gives up |

### 3.2 Component contracts

```python
class TalkingHeadRenderer(Protocol):
    def prepare_identity(self, reference_path: str) -> object: ...
    def start_session(self, identity: object) -> object: ...
    def push_audio(self, session: object, chunk: AudioChunk) -> None: ...
    def frames(self, session: object) -> Iterator[Frame]: ...
    def reset(self, session: object) -> None: ...
    def close_session(self, session: object) -> None: ...
```

`reset()` is the method that makes interruption possible: it drops whatever audio the
renderer has queued and whatever frames it has in flight, without tearing down the session
or the loaded weights. Everything else about interruption — deciding *when*, invalidating
in-flight artifacts, truncating history, switching the frame source — is deterministic
orchestration code outside this interface. The renderer has no concept of a turn, a
session state, VAD, or transport.

### 3.3 Measured results

**The headline numbers the brief asks for do not exist yet.** They require a talking-head
model, which requires the model spike, which requires a GPU:

| Metric | Measured | Hardware | Method |
|---|---|---|---|
| Audio-in to first-frame-out, real model | NOT YET MEASURED | | Blocked on the model spike |
| Steady-state fps, real model | NOT YET MEASURED | | Blocked on the model spike |
| Output resolution, real model | NOT YET MEASURED | | Blocked on the model spike |
| Peak VRAM | NOT YET MEASURED | | No GPU used yet |

#### 3.3.3 The full real stack — measured, every component live

`AVATAR_LLM=openai` (gpt-oss:20b via Ollama Cloud), `AVATAR_TTS=deepgram` (Aura-2),
`AVATAR_STT=deepgram` (Nova-3). All 17 end-to-end assertions pass, including barge-in with
stale-artifact drops verified from telemetry.

| Stage | Target §1.5 | Placeholders | **Real stack** | Verdict |
|---|---|---|---|---|
| End-of-turn detection | 100–300ms | 700ms | **700ms** | Config, not measurement. Over target by choice |
| LLM time-to-first-token | 200–500ms | 181ms (fake) | **1903–3234ms** | **4–6x over** |
| TTS time-to-first-audio | 100–300ms | 124ms (fake) | **949–1275ms** | **3–4x over** |
| Avatar first frame | 50–150ms | 404ms | **2983–4658ms** | Dominated by the two above |
| **Perceived total** | **<1000ms** | 430ms | **2992–4661ms** | **3–5x over** |

Adding the silence window, a full conversational turn measures **3.7s–5.4s** from the
candidate finishing their sentence to the avatar visibly answering.

**The headline finding for §3.4: not one of the three dominant terms is the renderer.**

1. **LLM TTFT (~1.9–3.2s)** is a free-tier cloud model. A paid low-latency model is the
   fix, and it is a spend decision rather than an engineering one.
2. **TTS TTFA (~0.9–1.3s)** is Aura over REST. Aura's WebSocket interface measured
   **351–361ms flat** with the connection cost paid once per session — a verified
   ~550ms saving, not yet implemented.
3. **End-of-turn (700ms)** is a policy number no hardware can improve.

A perfect, zero-latency talking-head model would still leave roughly **3.4s**. Any claim
that "more GPU" closes this gap is wrong, and the measurement is what makes that
falsifiable rather than an opinion.

#### 3.3.2 Real TTS versus the placeholder — measured A/B

Same pipeline, same host, one component swapped (`AVATAR_TTS`). Both runs passed all 17
end-to-end assertions, so this is a like-for-like comparison rather than two different
scenarios.

| Stage | Placeholder `ToneTTS` | Deepgram Aura-2 | Delta |
|---|---|---|---|
| `tts_first_audio` | 124ms | **893ms** | +769ms |
| `avatar_first_frame` | 404ms | **1226ms** | +822ms |
| `perceived_total` (to client paint) | 430ms | **1235ms** | +805ms |

**Real speech synthesis is the dominant term in the budget, and it is not close.** It
alone consumes more than the entire sub-second target the brief describes.

Adding the end-of-turn window, a full conversational turn measures roughly
**700ms + 1235ms ≈ 1.9s** from the candidate finishing their sentence to the avatar
visibly starting to answer — about twice the target. The two largest terms are the
silence window (a policy choice, see caveat 4) and network TTS.

Three things this changes about §3.4, and none of them are "buy a bigger GPU":

1. **The renderer is not the bottleneck.** Even a perfect zero-latency model leaves
   ~1.9s, because TTS and turn detection do not touch the GPU.
2. **Aura's warm/cold spread is ~400ms vs ~900ms**, and the first turn of every session
   pays the cold path. Connection pre-warming at session start is a cheap, unimplemented
   win worth roughly 500ms on turn one.
3. **The sub-second claim requires a streaming TTS**, not a request/response one. Aura's
   REST endpoint returns first audio only after synthesising a leading portion; the
   WebSocket interface exists precisely for this and is the obvious next measurement.

Measured on the host named in §3.3.1, with `AVATAR_LLM=scripted` — so `llm_ttft` (181ms)
is still a placeholder reporting its own configured delay, not a real model.

#### 3.3.1 What this measures — the session layer only, no ML model

These are real numbers from a real run, and they say nothing about any model. Read the
"what this actually measures" column before quoting any of them.

| Metric | Measured | Method | What this actually measures |
|---|---|---|---|
| Steady-state fps | **25.4** | 117 frames counted at a WebSocket client over 4.6s | The mixer's cadence under a real event loop. Target is 25. |
| Presentation timestamps | strictly monotonic, 40ms apart | every frame checked at the client | No gaps or duplicate pts across idle→renderer switches |
| Frames repeated | **0** | `FrameMixer.frames_repeated` | The stub never starved the mixer. A GPU renderer will. |
| Frames discarded on barge-in | **127** | `FrameMixer.frames_discarded` | Rendered frames for the cancelled turn, dropped |
| Turn start → first frame handed to mixer | **396ms** | `avatar_first_frame` histogram | Sum of the stubs' own configured delays plus real orchestration overhead — see below |
| Turn start → client reports paint | **398ms** | `perceived_total`, closed by a client `first_paint` report | The 2ms delta is loopback, not a browser. See the caveat. |
| Barge-in → server-side silence | **0.6ms** | `interrupt_to_silent` histogram | State transition, renderer reset, and flush dispatch. Excludes client-side stop. |
| Output resolution | 256×144 | BMP, uncompressed | Chosen to keep an unencoded 25fps stream tolerable, not for quality |
| End-of-turn detection | **700ms** | `turn_detect` histogram, driven by synthetic speech over a real socket | The configured silence window, by construction. See caveat 4 |

**Host:** Apple M1 Pro, 16GB, macOS 15.1.1, Python 3.12.3. **No GPU involved.**
Reproduce with `uvicorn avatar.server:app` then `python scripts/smoke_session.py`.

**Three caveats, each of which makes a number above less impressive than it looks:**

1. **`llm_ttft` (181ms) and `tts_first_audio` (122ms) are measuring their own
   configuration.** `ScriptedInterviewer` is told to wait 180ms before its first
   sentence and `ToneTTS` 120ms before its first chunk. The measurements confirm the
   instrumentation is wired to the right call sites; they are not evidence about any
   LLM or TTS engine. The same applies to the 396ms first-frame figure, which is
   dominated by those two delays plus the stub renderer's 200ms lookahead window.
2. **The 398ms "to paint" figure was closed by a Python client, not a browser.** The
   smoke script reports `first_paint` on receipt without decoding or rendering
   anything, so its 2ms delta over the server-side number is a lower bound on the
   real encode-decode-paint tail. `web/index.html` reports after `drawImage`, which is
   the honest measurement, and it has not been captured into this table yet.
3. **`interrupt_to_silent` (0.6ms) is server-side only.** It stops when the flush
   message is dispatched, not when the candidate stops hearing the avatar. The
   audible interruption latency includes the socket hop and the client stopping its
   scheduled buffers, and that number is not instrumented yet.
4. **`turn_detect` (700ms) is a configuration value, not a measurement.** It is the
   silence window the policy waits out before declaring a turn finished, and it appears
   in the budget unchanged no matter what hardware runs underneath. Recorded here
   because it occupies a real and large row in §1.5 — and worth saying plainly, because
   it is the one term in the whole budget that a faster GPU cannot touch. If a
   sub-second turnaround is the target, this single number is over two thirds of it.

### 3.4 Gap to production real-time

The gaps between this prototype and something production real-time. The GPU rows cannot be
quantified until the model spike produces real throughput figures, and are marked as such
rather than estimated.

| Gap | Cause | What closes it | Est. cost |
|---|---|---|---|
| **No lip-synced video at all** | No talking-head model integrated. The spike failed in setup before touching the GPU (§2.2.1) | A working spike run, then a renderer implementing the existing `TalkingHeadRenderer` Protocol | Renderer ~1 day once the spike lands. The spike itself is the unknown |
| **Turn-taking wait dominates the budget** | 700ms of deliberate silence before a turn is considered over. Over two thirds of a sub-second budget | Nothing hardware can do. Speculative execution — starting the LLM on a partial transcript and discarding it if the candidate resumes — is the real lever, and vendors publish that they do it | ~1 week, and it trades cost for latency: most speculative turns are thrown away |
| **TTS time-to-first-audio is 3–9× the target** | Aura over REST: one request per sentence, connection setup on the critical path | Aura's WebSocket interface. Measured at **351–361ms flat versus 907ms over REST** — verified, not estimated, and not yet implemented | ~1 day for ~550ms. The largest single unbuilt win |
| Mixer queue is unbounded | TTS runs ahead of playback and nothing applies backpressure. Observed: 127 frames queued and discarded on one barge-in, so a longer utterance queues proportionally more | Gate `push_audio` on `audio_sent_ms - audio_played_ms` exceeding a high-water mark, with a timeout so a silent client cannot stall the turn | ~0.5 day. Needs a fallback for clients that never acknowledge |
| Interruption latency measured server-side only | The sub-millisecond figure stops when the flush is dispatched, not when audio actually stops in the candidate's ear | Have the client report flush completion the way it already reports playback | ~2 hours |
| TCP head-of-line blocking, no congestion feedback | WebSocket over TCP. One lost packet stalls every frame behind it, and there is no congestion signal to adapt to | WebRTC. §1.4 states what the shortcut gives up | ~3 days including an SFU decision |
| Word-level truncation is estimated, not timed | The duration estimator assumes 150wpm rather than using real timings | Word timestamps from the TTS engine, which Aura exposes | Roughly free — a smaller change than the estimator it deletes |
| Renderer constructed per session | No pooling. Trivial for the stub; for a GPU model this is exactly the cold-start cost §1.4 argues cannot be paid at conversation start | Warm pool with session leasing | ~2 days, plus paying for idle GPU time |
| The detector under the turn policy is an energy gate | It distinguishes loud from quiet, not speech from noise. A loud enough cough will trigger it | Silero VAD, which is written and wired behind the same interface but **has never been executed** — no GPU-capable environment locally | ~1 day, mostly verification |

**Two of these are already measured rather than guessed**, which is the difference between
a gap list and a wish list: the Aura WebSocket win (351ms vs 907ms) and the queue depth
(127 frames on an observed barge-in). Both numbers came from runs, not from reasoning.

---

## 4. Build-vs-buy memo

### 4.1 Recommendation

**Keep the vendor for the rendering stage. Build the orchestration layer in-house, starting
now.** That is the hybrid, and the split is not a hedge — it falls directly out of what I
measured.

The single strongest reason: **I built a slice of this, and the model was never the
bottleneck.** A full turn measures 2.7–5.8s against a sub-second target, and the three
dominant terms are the end-of-turn policy, LLM time-to-first-token, and TTS. Set the
renderer to zero and roughly 2.6–5.7s remains. So the part a vendor sells is the part that
was never the problem, and the part that *is* the problem — turn-taking, cancellation,
history truncation, pipelining — is code the vendor's API does not write for you and cannot.

Building the renderer to replace a working vendor would be spending the scarcest engineering
capacity on the least-broken component.

### 4.2 Cost model

**Every figure below is an assumption, not a quote.** I have no vendor contract and did not
run a GPU, so the numbers are order-of-magnitude reasoning. They are stated so they can be
argued with, and the conclusion is deliberately insensitive to all of them except the last
row.

| Line item | Buy | Build |
|---|---|---|
| Per-minute marginal cost | Assume ~$0.10–0.30/min at list. Scales linearly and forever | GPU time only. A T4-class instance at ~$0.35–0.50/hr serving 2–3 concurrent sessions ≈ **$0.003–0.005/min** — one to two orders of magnitude lower |
| GPU capacity incl. idle in a warm pool | $0 — someone else's problem | **This is the line naive analyses omit.** §1.4 argues cold-loading at session start is unaffordable, so a warm pool is mandatory, and a warm pool means paying for idle GPUs. At interview-traffic burstiness, assume **50–70% idle**, so effective cost is 2–3× the figure above |
| Engineering to first production traffic | ~2–4 weeks of integration | **3–6 engineer-months.** The renderer is the small part. The rest: warm pooling with session leasing, WebRTC transport, GPU autoscaling, per-session isolation, a quality bar measurable without a human, and the shadow-mode comparison harness in §5 |
| Ongoing engineering + on-call | Near zero. Vendor absorbs model upgrades, CUDA drift, capacity | **0.5–1 FTE indefinitely.** A pager that did not exist before, plus a model that ages and a CUDA/TensorRT stack that moves underneath it |
| Break-even volume | — | Marginal cost favours building almost immediately. **Fully loaded, break-even is somewhere above ~50,000–100,000 avatar-minutes/month sustained** — and the engineering line dominates so heavily that the exact per-minute rate barely moves it |

**The honest reading of this table:** the per-minute comparison flatters building by one to
two orders of magnitude and is *the wrong number to decide on*. At startup volume the
engineering and on-call lines dwarf the entire vendor bill. If the current spend is a few
thousand dollars a month, building cannot pay for itself on cost — an engineer-month costs
more than a year of the vendor.

**So a cost-driven "build" is only credible above roughly 50–100k minutes/month sustained.**
Below that, anyone arguing to build on cost grounds has not costed the engineering.

### 4.3 Non-cost factors

Cost does not favour building at startup volume, so if the answer is ever "build", the reason
lives here. Weighed honestly, including where the vendor wins.

| Factor | Weight | Assessment |
|---|---|---|
| **Data residency** | **Highest** | Candidate audio and video leaving our infrastructure is the one factor that can force this decision regardless of cost. Interview recordings are sensitive personal data, and a single enterprise client with a contractual residency requirement converts this from an optimisation into a blocker. **This is the strongest build argument and it is not economic.** |
| Latency control | High, but **not** in the vendor's favour or against it | I measured this and it changed my mind mid-analysis. The terms I would need to attack are STT, LLM, TTS and the turn policy — **all of which I already control** in the hybrid. Owning the renderer would buy control over the term that is not the problem |
| Customisation | Medium | Persona control, turn-taking behaviour, interview-specific interruption policy. Almost all of it lives in orchestration, which the hybrid already owns. Genuinely renderer-specific customisation is a narrow set |
| Vendor concentration risk | Medium | A single vendor for a core product surface is real exposure — pricing power, roadmap divergence, acquisition, shutdown. Mitigated substantially by the boundary in §3.2: the interface exists, so the switching cost is bounded and known rather than open-ended |
| **Visual fidelity** | **Where the vendor is genuinely better** | Stated plainly: a funded team iterating full-time on one model will beat what I can build, and MuseTalk at a 256×256 face region is not close to a vendor's output. The brief puts fidelity out of scope for the *prototype*; it is emphatically not out of scope for a product a candidate is judged through |
| Time to market | Vendor | Weeks against engineer-months. If the avatar channel is still being validated commercially, building first is optimising a bet not yet won |

### 4.4 What would change my mind

Numeric, and each one is checkable rather than a feeling.

| Trigger | Threshold | Direction |
|---|---|---|
| **Volume** | Above **75,000 avatar-minutes/month sustained for two consecutive quarters** | → **Build the renderer.** Two quarters, because a single spike is a seasonal hiring cycle, not a trend |
| **Contractual data-residency requirement** | **Any single signed enterprise client** requiring candidate media to stay in our infrastructure or a named region | → **Build immediately, regardless of volume.** This is the one trigger that overrides the cost model entirely |
| **Vendor p95 latency** | p95 utterance-to-utterance above **1.5s**, measured by us on our traffic, sustained a month | → Build. Note the emphasis: **measured by us.** The published ~600ms is a marketing figure I have not verified |
| **Vendor pricing change** | Any increase above **30%**, or a move to a model that penalises our burst pattern | → Re-run §4.2 with real quotes. A 30% rise on a small bill is still a small bill |
| **Vendor viability event** | Acquisition, a funding event implying a strategy change, or two P1 incidents in a quarter | → Activate the §5 shadow-mode plan as a **contingency**, whether or not we intend to cut over. The escape hatch has to be tested before it is needed |
| **A model clears both gates with a trivial runtime** | Published real-time on a T4, commercial weights, no compiled runtime, and it installs from a clean clone | → Revisit. The 3–6 engineer-month estimate is dominated by serving infrastructure, but a genuinely drop-in model moves the renderer line enough to re-run the case |

### 4.5 Risks in my own recommendation

Three ways I could be wrong, in the order I think they are most likely.

**1. I may be over-weighting my own measurements.** My latency numbers come from free-tier
infrastructure — `gpt-oss:20b` on Ollama Cloud's free tier produced an LLM time-to-first-token
between 1.6 and 4.7 seconds. A paid low-latency endpoint plausibly cuts that to 300–500ms.
If every non-renderer term shrinks that far, the renderer becomes a much larger fraction of
the remaining budget, and "the model was never the bottleneck" weakens considerably. **My
central claim is measured, but it is measured on the cheapest possible stack**, and that is a
real threat to it.

**2. The hybrid may be the worst of both worlds operationally.** I have argued the split is
clean because the boundary is clean — but we would be running the orchestration on-call
burden *and* paying vendor per-minute rates, with a network hop between the two adding
latency I have not measured. A vendor's integrated pipeline may beat a split one precisely
because it is not split. I have no measurement either way, and that is a gap in this
recommendation rather than a point in its favour.

**3. Sunk cost is pushing me toward "build" more than I would like.** I spent a day and a
half on the orchestration layer and I am recommending keeping it. That is exactly the bias
this exercise is testing. The check I applied: would I recommend building the orchestration
if I had *not* written it? I think yes — because the alternative is accepting a vendor's
turn-taking policy and interruption semantics, and interruption behaviour is product-defining
for interviews. But I hold that less confidently than the rest of this memo, and someone
should push on it.

**One thing I would want before committing either way**, and which I could not do here: a
measured p95 of the vendor's actual latency on our own traffic, and a real quote at our real
volume. Both of the load-bearing numbers in §4.2 are assumptions, and §5's preconditions put
observability on the vendor path first precisely so that this decision gets made on measured
data rather than on this memo.

---

## 5. Migration plan

Written as a runbook rather than a narrative, against the standard the brief sets: another
senior engineer should be able to execute this without me in the room.

**Assumes §4 concluded "build" or "hybrid."** If §4 concludes "keep the vendor," this plan
is the contingency, not the roadmap.

### 5.1 Preconditions

Every one of these must be true before shadow mode starts. The ordering is deliberate:
**observability goes onto the vendor path first**, because without a baseline measured the
same way, phase 1 produces two sets of numbers that cannot be compared.

| # | Precondition | Why it gates everything after it |
|---|---|---|
| 1 | The §1.7 telemetry is deployed **on the vendor path**, in production, for at least two weeks | This is the baseline. Comparing in-house p95 against a vendor number from a marketing page is not a comparison |
| 2 | Both paths sit behind one internal interface, with the vendor as an implementation of it | If the vendor call is spread through the codebase, there is nothing to flag. This is the `TalkingHeadRenderer` boundary generalised to the whole avatar stage |
| 3 | A quality bar is written down and is **measurable without a human in the loop** | "Looks fine" cannot gate an automated rollout. Lip-sync offset in ms and frame-drop rate can |
| 4 | Rollback is a config flip, tested at least once in staging under live traffic | An untested rollback is a hope. See §5.4 |
| 5 | Warm-pool capacity exists for the target cohort **plus headroom**, with a documented cold-start cost | Discovering the pool is undersized during a live interview is the worst time to learn it |
| 6 | An incident owner and an escalation path exist for the in-house path | The vendor absorbed this operational load. Someone now carries a pager who did not before |

Precondition 3 is the one most often skipped and the most expensive to skip: without a
machine-checkable quality bar, every promotion decision becomes a meeting.

### 5.2 Phase 1 — Shadow mode

| | |
|---|---|
| Traffic served by | **Vendor.** Candidates see only vendor output |
| In-house path | Renders the same session in parallel; **output discarded, never shown** |
| Duration | **2–4 weeks**, and long enough to include at least one full business cycle plus one deploy of each path |
| Compared | Latency distributions (p50/p95/p99, not means), failure and reconnect rates, lip-sync offset, frame-drop rate, cost per avatar-minute |
| Exit criteria | In-house **p95** first-frame latency within an agreed margin of vendor p95; failure rate no worse; quality bar (precondition 3) met on ≥99% of shadowed sessions |
| Cost | **Double render cost for the whole window, plus the shadow GPU pool.** State it in the budget request up front — it is the single largest line in this plan and discovering it late kills the project's credibility |

**Why percentiles, not means:** a mean hides exactly the failure this system is prone to.
One session in fifty with a 6-second first frame is a candidate having a visibly broken
interview, and it barely moves an average.

**Two things shadow mode cannot tell you**, and they must be stated rather than assumed
away:

- **Nothing about perceived quality.** No candidate is watching the shadow output. Only
  phase 2 tests that.
- **Nothing about interaction dynamics.** Barge-in behaviour depends on the candidate
  reacting to what they hear. Shadowed output nobody hears cannot produce a realistic
  interruption pattern, so the interruption path is genuinely only exercised from stage 1
  of phase 2 onward.

### 5.3 Phase 2 — Flagged rollout

Cohorts are ordered by **how survivable a bad outcome is**, not by size. A candidate
interview is a one-shot event with real consequences for the candidate — a broken one cannot
be retried away — so real candidates come last, and internal users absorb the first
failures.

| Stage | Cohort | Duration | Promote if | Abort if |
|---|---|---|---|---|
| 1 | **Internal only** — employees running practice interviews | 1 week | Zero P1 incidents; quality bar met; the on-call rotation has handled at least one real alert | Any P1, or the quality bar missed on >1% of sessions |
| 2 | **5% of real candidates**, excluding any client with a contractual SLA | 1 week | p95 latency and failure rate within the phase-1 margin on real traffic; no increase in interview-abandonment rate | Abandonment rate up at all, or any candidate-reported failure traced to the render path |
| 3 | **25%**, still excluding SLA-bound clients | 2 weeks | Metrics hold at 5× the traffic; warm pool holds p95 at peak concurrency | Pool saturation, or p95 degrading as concurrency rises — that is a capacity problem, not a model problem |
| 4 | **100%**, SLA clients included, vendor kept warm | 2 weeks | All of the above sustained through a peak period | Anything above |

**Abandonment rate is the metric to watch above all others.** Latency percentiles say
whether the system is fast; abandonment says whether the experience is acceptable, and it
is the one number a candidate votes with.

**One explicit carve-out:** clients with contractual latency or availability SLAs stay on
the vendor until stage 4. Migrating them early puts a commercial commitment at risk to save
a week of rollout.

### 5.4 Rollback

**Rollback is a config flip, not a deploy.** If rolling back requires a build, the mean
time to recovery is however long CI takes, and that is not a rollback plan.

| | |
|---|---|
| Mechanism | Flip the routing flag to `vendor`. Same interface (precondition 2), so no code path changes |
| Target time-to-rollback | **< 60 seconds** from decision to all new sessions on the vendor |
| Who may execute | **Any on-call engineer, without approval.** A rollback needing a manager's sign-off will be delayed past the point of usefulness |
| Trigger | Explicit and pre-agreed: any P1, p95 breaching the agreed ceiling for 5 consecutive minutes, or the quality bar failing on >1% of sessions in a 15-minute window |
| Verification | The flag's effect is visible in the telemetry within one session's length; the dashboard shows path attribution per session |

**Sessions in flight at the moment of rollback** — the detail that separates a plan from a
hope. There are two options and they must be chosen deliberately, not discovered:

- **Let them finish on the in-house path.** An interview is stateful — conversation
  history, a warm renderer session, the candidate mid-sentence. Cutting over mid-session
  means a visible discontinuity: the avatar's appearance changes between one turn and the
  next, which is worse for that candidate than a slightly degraded render.
- **Hard-cut them to the vendor.** Correct only when the failure is severe enough that
  finishing is worse than a visible seam — the renderer producing garbage frames, or
  leaking audio between sessions.

**Default: drain, don't cut.** New sessions go to the vendor immediately; in-flight
sessions finish where they started. The hard cut is reserved for correctness and privacy
failures, and it needs its own separate flag so that reaching for it is a deliberate act.

Maximum drain time is bounded by the interview length cap, so it must be stated: with a
60-minute cap, full drain is up to 60 minutes. If that is unacceptable, the interview cap
is the thing to change, not the rollback design.

### 5.5 Decommission

Do **not** cancel the vendor contract at 100% traffic. Keep the integration warm — flag
intact, credentials valid, path tested — for **at least one full billing cycle plus one
quarter** of stable 100% operation.

| Milestone | Gate |
|---|---|
| Stop shadow rendering | 2 weeks after 100%, when the comparison has served its purpose |
| Reduce vendor spend to a minimum retainer | One quarter at 100% with no rollback exercised |
| Cancel the contract | Two quarters stable, **and** the in-house path has survived one GPU-provider incident, one model upgrade, and one peak period |
| Delete the integration code | Never, until the contract is cancelled. Dead code that is a working escape hatch is worth its maintenance |

**Run the vendor path in staging on a schedule even after cutover**, so the escape hatch is
known-good rather than assumed-good. An untested fallback discovered to be broken during an
incident is equivalent to having no fallback, and the failure mode is silent — credentials
expire, APIs version, and nothing tells you until you need it.

---

## 6. Sources

Numbered so §1's **[C]** tags can reference them. Primary and secondary are separated
deliberately: a vendor's own documentation and a published paper are primary; a
comparison blog post is not, and citing one as though it were would undermine the whole
confirmed-vs-inferred framing this document rests on.

**A note on what these sources can and cannot support.** Every source below is either
open-source documentation or a vendor's own published material. **No source here is inside
information about any vendor's implementation.** Where §1 describes vendor internals, the
source supports an *open-source analogue* or a vendor's own *performance claim* — never
the vendor's actual architecture. That gap is exactly what the **[I]** tag is for, and it
is the reason the tagging is not a formality.

**Primary — open-source implementations (code and documentation read directly)**

1. **MuseTalk** — <https://github.com/TMElyralab/MuseTalk>
   Verified directly from the repository: trained in the latent space of `ft-mse-vae`;
   audio encoded by a **frozen `whisper-tiny`**; generation network borrowed from the
   **Stable Diffusion v1.4 U-Net** with audio fused to image embeddings by
   **cross-attention**; **"NOT a diffusion model"** — single-step latent inpainting;
   **256×256 face region**; claims **"30fps+ on an NVIDIA Tesla V100"**. Licence: *"The
   code of MuseTalk is released under the MIT License. There is no limitation for both
   academic and commercial usage"*, and *"The trained model are available for any purpose,
   even commercially."* Supports §1.2 and §1.3.
2. **MuseTalk paper** — <https://arxiv.org/abs/2410.10122>
   Multi-scale feature fusion within the U-Net; the SIS and AAM strategies.
3. **Ditto** — <https://github.com/antgroup/ditto-talkinghead> (Ant Group, ACM MM 2025)
   Motion-space diffusion; **denoising steps reduced from 50 to 10** with comparable
   evaluated quality; **DiT converted to TensorRT**; jointly optimised for *"streaming
   processing, real-time inference, and low first-frame delay."* Apache-2.0. Supports
   §1.3's surviving row and §2.2's operability argument.
4. **Ditto paper** — <https://arxiv.org/abs/2411.19509>
5. **Wav2Lip** — <https://github.com/Rudrabha/Wav2Lip>
   Licence verified directly, and it is the basis for §2.2's rejection-on-licence:
   *"This repository can only be used for personal/research/non-commercial purposes"*, and
   *"As the models are trained on the LRS2 dataset, any form of commercial use is strictly
   prohibited."* No formal OSS licence is offered.

**Primary — vendor's own published claims (performance, not architecture)**

6. **Tavus — Conversational Video Interface overview** —
   <https://docs.tavus.io/sections/conversational-video-interface/overview-cvi>
   The published pipeline order: perception (Raven) → conversational flow (Sparrow) → STT →
   LLM → TTS → realtime replica (Phoenix). Named **WebRTC** transport. Supports §1.1's
   stage ordering and §1.4's transport argument.
7. **Tavus — "the world's fastest Conversational Video Interface"** —
   <https://www.tavus.io/post/introducing-the-worlds-fastest-conversational-video-interface-for-developers>
   Claims **~600ms utterance-to-utterance**, and under one second generally. This is the
   number §1.5's target column is calibrated against, and the number §3.3.3's measured
   3.7–5.4s should be read next to. **It is a vendor marketing claim, not an independent
   measurement** — worth stating whenever it is cited.
8. **Tavus — turn-taking and speculative execution** —
   <https://www.tavus.io/blog/generative-ai-avatars-table-stakes-conversation-leap>
   Describes retrieval on every utterance without a perceptible pause, and **speculative
   execution** — preparing a response while the user is still speaking. Relevant to §3.4:
   it is a way to attack the end-of-turn term that this prototype does not implement.

**Secondary / context — used for orientation only, never as evidence for a [C] tag**

9. Comparison write-ups of open-source lip-sync models (relative speed and quality
   rankings of LivePortrait, SadTalker, Wav2Lip, MuseTalk). Directional only; the specific
   figures in §2.2 come from the repositories in 1–5, not from these.
10. Assorted engineering blog posts on WebRTC versus WebSocket for low-latency media. Used
    to frame §1.4's transport reasoning; the argument there stands on the protocols'
    documented properties rather than on any of these posts.

**Explicitly not consulted, and the resulting [U]:** no vendor's private documentation, no
network-level inspection of a live vendor session, and no vendor account of any kind. The
brief states none is required. The practical consequence is that §1.3's sub-claim A
(mouth-region synthesis composited onto replayed body footage) **remains inferred** — the
verification methods listed there are the ones that *would* confirm it, and they have not
been carried out against a live vendor session.
