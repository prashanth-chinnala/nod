# PROCESS.md

Real-time conversational avatar — architecture research, prototype, and build-vs-buy recommendation.

| | |
|---|---|
| Author | *your name* |
| Repository | <https://github.com/prashanth-chinnala/nod> |
| Time spent | *actual hours, honestly* |
| Hardware used | *e.g. Colab T4 16GB / RTX 3090 / CPU-only M2* |
| Prototype status | Real spoken conversation working end to end. No talking-head model — see §3.1 |

> **Delete every blockquote like this one before submitting.** They are drafting prompts, not content.

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

> Include a real **[U]** or two. A document with zero unknowns reads as either incurious or dishonest, and §7.3 grades exactly this.

**`[HUMAN]` — the tagging itself is the candidate's honesty claim about what they personally
verified. An agent must not assign these tags.**

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
| Input to enrollment | A short reference video or a single image of a face, roughly front-on. MuseTalk operates on a **256×256 face region** cropped from the frame, so framing matters more than duration | MuseTalk README | |
| Artifact produced | Per-frame VAE latents of the reference frames, plus face detection / parsing / bounding-box metadata. **No per-person model weights** | MuseTalk README + inference code | |
| Enrollment latency | Dominated by face detection and VAE encoding over the reference frames — seconds to minutes depending on clip length. `NOT YET MEASURED` here (blocked on M0) | — | |
| Reusable across sessions? | Yes. The artifact is a function of the reference clip alone, so it is computed once per persona and cached | Follows from the above | |
| Per-person GPU state at inference | Only the cached latents and crop metadata. The U-Net, VAE, and audio encoder are shared across every persona on the worker | MuseTalk README | |

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

| Class | Example work | Why it survives or dies |
|---|---|---|
| Full generative video diffusion | Sora-class, Stable Video Diffusion | **Dies on throughput by orders of magnitude.** Generating a whole talking human per frame solves a vastly larger problem than needed, and nothing in this class runs at 25fps. Also uncontrollable: no guarantee the identity stays stable across a 20-minute interview |
| Audio-conditioned latent diffusion, multi-step | Diffusion talking-head research generally | **Dies on the step count.** *N* denoising steps per frame multiplies per-frame cost by *N*. At 25fps there is a ~40ms budget per frame; multi-step diffusion is nowhere near it |
| Motion-space diffusion + neural render | **Ditto** (Ant Group, ACM MM 2025) | **Survives, with an operational cost.** Diffuses in a compact *motion* space rather than pixel space, cuts denoising from **50 steps to 10**, and compiles the DiT to **TensorRT**. Explicitly built for streaming and low first-frame delay. The TensorRT engines are GPU-specific, which is a real deployment constraint |
| Latent-space mouth inpainting, single-step | **MuseTalk** | **Survives, and is the cheapest.** Borrows the Stable Diffusion v1.4 U-Net architecture but **is not a diffusion model** — it inpaints the mouth region in VAE latent space in a **single step**. Claims **30fps+ on a V100** at a 256×256 face region |
| 3D rig driven by visemes | Classical game-engine avatars | **Survives on latency, dies on realism.** Viseme-driven rigs are trivially real-time and fully controllable, but look animated rather than photographic — the wrong product for an interview meant to feel human |
| 3D Gaussian splatting / NeRF per-identity | Per-identity neural head research | **Dies on enrollment, not inference.** Inference can be fast, but each identity needs its own trained representation — minutes to hours of per-person GPU work. That breaks the "upload a reference clip, interview in seconds" product shape and reintroduces the per-person-weights cost §1.2 rejected |

**What survives** is a narrow band: *single- or few-step generation, in a compact latent or
motion space, over a small region of the frame, conditioned on precomputed identity
features.* That is a much smaller problem than "generate video of a person talking," and
recognising that it is smaller is the actual insight.

#### 1.3.1 Two sub-claims that separate a surface answer from a real one

**Sub-claim A — most of the frame is replayed, not generated.**

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
not raw waveform or bare mel spectrograms.**

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

**Session and model pooling.** *Cold-start cost of loading model weights + CUDA context, why that cannot be paid at conversation start, what a warm pool implies for cost (idle GPU time you pay for), how sessions bind to workers, and what happens under a thundering herd of simultaneous interviews.*

**Frame transport.** *WebRTC vs. WebSocket vs. HLS/DASH. Kill HLS on segment granularity. Justify WebRTC on jitter buffering, congestion control, NAT traversal, and browser-native decode. Note whether an SFU sits in the path and what that adds in latency and operational surface.*

**Audio/video sync.** *Whose clock wins, and how lip-sync survives network jitter. If audio and video travel as separate tracks, what keeps them aligned?*

**The pipelining insight.** *State plainly that the stages must overlap: LLM tokens stream to sentence-chunked TTS, TTS audio chunks stream to the renderer, the first video frame emits after the first ~200ms of audio exists rather than after full synthesis. Sequential-and-complete execution of the same stages produces a multi-second turnaround with identical component performance.*

### 1.5 Latency budget

> Fill the "target" column with your reasoned budget for a sub-second perceived turnaround, and the "prototype" column with what you actually measured. The gap between those two columns is what §7.6 grades.
>
> The ranges below are starting points from my own reasoning, not sourced figures — replace them with your own and tag accordingly.

| Stage | Target (ms) | Measured in prototype (ms) | Tag | Notes |
|---|---|---|---|---|
| End-of-turn detection | 100–300 | **700** | | Configuration, not measurement. Over twice the top of my own target — see §3.3.1 caveat 4 |
| Speech-to-text finalize | 50–150 | NOT YET MEASURED | | Streaming, so mostly already done incrementally |
| LLM time-to-first-token | 200–500 | NOT YET MEASURED | | Only TTFT matters, not total generation |
| TTS time-to-first-audio | 100–300 | **893** cold / **~400** warm | | Deepgram Aura-2, real. 3–9x over my target. See §3.3.2 |
| Avatar first frame | 50–150 | NOT YET MEASURED | | |
| Encode + network + jitter buffer | 50–150 | NOT YET MEASURED | | |
| **Perceived total** | | NOT YET MEASURED | | |

> Add a sentence on **which term you would attack first** if handed a latency budget overrun in production, and why. That single sentence is a leadership signal.

### 1.6 Failure and edge handling

**Interruption (barge-in).**

> The cancellation chain: VAD fires → cancel LLM generation → cancel TTS → flush frame buffer → transition avatar to a listening state. Give the target time from user speech onset to avatar going quiet.
>
> Then the part most candidates miss: **conversation history must be truncated to what the candidate actually heard, not what was generated.** If the LLM's context claims it asked a question the candidate never heard, every subsequent turn is subtly wrong. Explain how you'd know the truncation point (audio frames actually acknowledged as played, not frames sent).

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

| Criterion | Weight | Why it matters here |
|---|---|---|
| Achievable fps on accessible hardware | | Real-time is the point |
| First-frame latency | | Distinct from throughput — a model can hit 30fps and still have a slow first frame |
| License, code **and** weights | | These differ, often materially |
| Streaming-native vs. batch | | A batch model cannot be made streaming inside this time-box |
| Output quality at the target resolution | | Bounded — §4 puts fidelity out of scope |
| Maintenance health | | Commits, issue response, releases |
| Setup fragility | | §6 requires a clean-clone build. **Evidence: MuseTalk's `download_weights.sh` and `pip install` both exited 0 without installing the model — see §2.2.1** |

### 2.2 Candidates evaluated

| Model | fps / latency | License (code / weights) | Streaming? | Verdict |
|---|---|---|---|---|
| MuseTalk (`0a89dec`) | **NOT YET MEASURED** — spike run 1 failed in setup before touching the GPU | MIT code; weights permit commercial use | Yes, documented realtime mode | Undecided. See §2.2.1 |
| Ditto | NOT YET MEASURED — not attempted | Apache-2.0 | Yes, streaming-native | Undecided. TensorRT 8.6.1 with GPU-specific prebuilt engines fights an ephemeral Colab runtime |
| Wav2Lip | Published ~real-time on modest GPUs | **Licence prohibits commercial use** | No | **Rejected on licence.** Not run |
| LatentSync | ~10x slower than real time (~100s for 10s of video on a 4090, published) | Open | No | **Rejected on latency.** Not run |

#### 2.2.1 Spike run 1 — MuseTalk on a free-tier Colab T4

**Outcome: failed in setup. No inference occurred, so nothing about this model's
throughput has been measured.** Full post-mortem in [`docs/M0_SPIKE.md`](docs/M0_SPIKE.md) §4.

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

> Name the pick, the single decisive criterion, and — importantly — the strongest argument against it. Then state what would make you switch.

**`[HUMAN]`** — the selection is a judgment call with a memo attached. Agent may assemble
evidence into §2.2; the pick and its rationale are the candidate's.

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
| Session start/stop lifecycle | **Built** (M1) | State machine complete; 131 tests, all GPU-free |
| Idle-loop fallback frame | **Built** (M1, M3) | `IdleLoop` + `FrameMixer`. The clip is a synthetic pulse, not a face — a real one needs M4's preparation script |
| Interruption handling | **Built** (M1, M3) | Turn-epoch cancellation, verified end-to-end including the client-side audio flush |
| Browser streaming transport | **Built** (M3) | WebSocket, the shortcut the brief permits. Costs stated in `transport/websocket.py` |
| Browser client | **Built** (M3) | Canvas + Web Audio + mic, plain JS, no build step |
| Playback acknowledgement | **Built** (M3) | Client reports how much audio actually played, including partial buffers stopped by a barge-in |
| End-to-end latency to browser paint | **Built** (M3) | Client reports first paint; the server cannot measure this for itself |
| **Audio in → lip-synced video out** | **Not built** | Blocked on M0. Needs a GPU, and Rule 1 forbids estimating what it would do |
| **A talking-head model of any kind** | **Not built** | Same. `StubRenderer` proves the interface, not the capability |
| Real STT | **Built** | Deepgram Nova over a persistent WebSocket. Transcribes continuously; the local turn policy decides when the turn ends, not the vendor's endpointing — see the `Transcriber` docstring |
| Real TTS | **Built** | Deepgram Aura-2. `container=none` is load-bearing: the default response carries a 44-byte RIFF header that would be played as PCM |
| Real LLM | **Built** | Two adapters (Anthropic, OpenAI) behind one `SentenceStream`. `OPENAI_BASE_URL` also reaches Ollama / LM Studio / vLLM, so a local model needs no new adapter |
| Configuration | **Built** | `.env.development` / `.env.local` / `.env` layered at import, real env vars winning; `GET /config` reports what each boundary resolved to and which files were read. Every default is a working no-credential one |
| Turn-taking policy | **Built** (M4) | Server-side. Onset, hysteresis, retraction, and end-of-turn as separately tuned decisions; 30 tests over probability sequences |
| **A real voice activity detector** | **Partly** (M4) | The policy is real and tested. The default detector under it is an energy gate that cannot tell speech from a door. `SileroVad` is written, wired, and **has never been executed** — no torch in the dev environment |
| Frame encoding | **Built** | PNG, stdlib `zlib`, no new dependency. **108.10 KB → 0.57 KB per frame, 22.20 → 0.12 Mbps at 25fps — 188×.** Was the reason 0.5fps of 25 arrived through a tunnel. The client sniffs the format from magic bytes, so M2 can switch to JPEG for photographic frames with no protocol change |
| Client jitter buffer | **Built** | 150ms lead, `?audioLead=` overridable, underruns counted and surfaced. Absent before, which is why audio was clean on localhost and broke through a tunnel |
| Warm model pooling | Deferred (M7) | Described in §1.4. Constructing a renderer per session is exactly the cost that section argues cannot be paid at conversation start |
| Multi-session concurrency | Deferred (M7) | One orchestrator per socket is wired and works; only one session has been exercised |
| WebRTC transport | Deferred (M7) | Stretch goal. §1.4 states what the WebSocket shortcut gives up |

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
model, which requires M0, which requires a GPU:

| Metric | Measured | Hardware | Method |
|---|---|---|---|
| Audio-in to first-frame-out, real model | NOT YET MEASURED | | Blocked on M0 |
| Steady-state fps, real model | NOT YET MEASURED | | Blocked on M0 |
| Output resolution, real model | NOT YET MEASURED | | Blocked on M0 |
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

#### 3.3.1 What M3 does measure — session layer only, no ML model

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

Gaps identified from the M3 run. The GPU rows cannot be quantified until M0 produces
real throughput figures, and are marked as such rather than estimated.

| Gap | Cause | What closes it | Est. cost |
|---|---|---|---|
| No lip-synced video at all | No model selected or integrated | M0 spike then M2 | `[HUMAN]` — depends on the model pick |
| Uncompressed frames, ~2.7MB/s at 256×144 | No encoder; BMP was chosen to avoid an image dependency | JPEG or WebP from the renderer's own output in M2 | ~1 day; roughly a 40× bandwidth reduction |
| Mixer queue is unbounded | The TTS runs 4× ahead of playback and nothing applies backpressure. Observed directly: 127 frames were queued and discarded on one barge-in, so a longer utterance queues proportionally more | Gate `push_audio` on `audio_sent_ms - audio_played_ms` exceeding a high-water mark, with a timeout so a silent client cannot stall the turn | ~0.5 day. Needs a fallback for clients that never acknowledge |
| Interruption latency measured server-side only | The 0.6ms figure stops when the flush is dispatched, not when audio actually stops | Client reports flush completion the way it already reports playback | ~2 hours |
| No jitter buffer, no congestion feedback | WebSocket over TCP. Head-of-line blocking turns one lost packet into a stall for every frame behind it | WebRTC via aiortc (M7) | ~3 days including an SFU decision |
| Word-level truncation is estimated, not timed | `estimate_duration_ms` assumes 150wpm | Word timestamps from a real TTS engine, which most expose | Free once M4 lands; it is a smaller change than the estimator it deletes |
| Renderer constructed per session | No pooling. Trivial for the stub; for a GPU model this is the cold-start cost §1.4 argues cannot be paid at conversation start | Warm pool with session leasing (M7) | ~2 days, plus paying for idle GPU time |
| Turn detection is a client-side energy gate | No VAD | Silero VAD with separate onset and end-of-turn thresholds (M4) | ~1 day |

---

## 4. Build-vs-buy memo

> Write this **last**, and write it as though you had not just spent two weeks building a prototype. Sunk cost is the bias being tested.

**`[HUMAN]` — this entire section. Do not let an agent draft the recommendation or the
thresholds.**

### 4.1 Recommendation

> One sentence, up front. Keep vendor / build in-house / hybrid.

### 4.2 Cost model

| Line item | Buy | Build |
|---|---|---|
| Per-minute marginal cost | | |
| GPU capacity (incl. idle in warm pool) | | |
| Engineering to first production traffic | | |
| Ongoing engineering + on-call | | |
| Break-even volume | | |

> State your assumptions as assumptions. The engineering line usually dominates by an order of magnitude and is the line naive analyses omit.

### 4.3 Non-cost factors

> Cost rarely favors building at startup volume. The credible build arguments are elsewhere, and §1 of the brief names them: data residency (candidate audio/video leaving your infrastructure), latency control, customization, vendor concentration risk. Weigh each honestly — including where the vendor is genuinely better, such as visual fidelity you cannot match in-house.

### 4.4 What would change my mind

| Trigger | Threshold | Direction |
|---|---|---|
| Volume | | |
| Contractual data-residency requirement | | |
| Vendor p95 latency | | |
| Vendor pricing change | | |
| Vendor viability event | | |

> Numeric thresholds. "If we grow a lot" is not a trigger; "above N,000 avatar-minutes/month sustained for two quarters" is.

### 4.5 Risks in my own recommendation

> Name the two or three ways you could be wrong. §7.3 rewards this directly.

---

## 5. Migration plan

Written as a runbook rather than a narrative, against the §8 test: another senior engineer
should be able to execute this without the author in the room.

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
