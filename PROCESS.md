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

> One paragraph, then one diagram. The diagram should show the per-turn loop: candidate audio in → turn detection → STT → LLM → TTS → avatar render → transport → browser. Mark on it which stages are streaming and which are blocking — that distinction is the core insight of the whole document.

### 1.2 Identity capture

> Answer: what artifact does a reference video become, and what does that imply for serving?
>
> Cover: input requirements (duration, framing), whether processing is offline/one-time or per-session, and — the load-bearing question — **per-person weights vs. shared model + identity embedding**. State the serving consequence explicitly: per-person weights mean cold-loading a checkpoint onto a GPU worker at session start; a shared model with cached identity features does not.
>
> Note whether any *precomputation* step exists (face detection, parsing, appearance-feature extraction, latent encoding of reference frames). If it does, that is architecturally significant: it is the reason first-frame latency can be low at conversation time.

| Question | Answer | Tag |
|---|---|---|
| Input to enrollment | | |
| Artifact produced | | |
| Enrollment latency | | |
| Reusable across sessions? | | |
| Per-person GPU state at inference | | |

### 1.3 The audio-to-video mechanism

> This is the highest-value section. Structure it as **elimination**, not description — the latency constraint rules out most of the design space, and showing that reasoning is worth more than naming the right answer.
>
> Walk the candidate model classes and kill them one by one against the ~25–30fps / sub-second first-frame requirement:

| Class | Example work | Why it survives or dies |
|---|---|---|
| Full generative video diffusion | | |
| Audio-conditioned latent diffusion, multi-step | | |
| Motion-space diffusion + neural render | | |
| Latent-space mouth inpainting, few-step | | |
| 3D rig driven by visemes | | |
| 3D Gaussian splatting / NeRF per-identity | | |

> Then state what you believe production systems actually run, and why. Two sub-claims worth making explicitly because they are the difference between a surface answer and a real one:
>
> 1. **What is actually generated vs. replayed.** If only the mouth/lower-face region is synthesized and composited onto pre-recorded body footage, the model is solving a far smaller problem than "generate a talking human." Say so, and say how you'd verify it from observed output (fixed background? repeating gesture cycles? seam artifacts at the jaw?).
> 2. **The audio conditioning signal.** Raw waveform, mel spectrogram, or features from a self-supervised speech encoder (Whisper encoder / HuBERT / wav2vec2)? Published real-time open-source systems use the latter; that is a strong **[C]** anchor for an **[I]** claim about vendors.
>
> Also address: how is temporal consistency maintained across chunk boundaries when frames are generated in a streaming fashion? This is where naive implementations visibly break.

### 1.4 Real-time serving architecture

> Four sub-questions. Be concrete about numbers even when they're inferred.

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

> §7.5 makes this mandatory, and it's cheap to do well. For each stage in the §1.5 table, specify the instrumentation.

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

> Add one line on **trace propagation**: a single trace ID following one conversational turn across STT, LLM, TTS, render, and transport. Without it, "the avatar felt laggy" is unfalsifiable.

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

> Include at least one model you rejected **on license** and one you rejected **on latency**. That demonstrates the criteria were real rather than decorative.

#### 2.2.1 Spike run 1 — MuseTalk on a free-tier Colab T4

**Outcome: failed in setup. No inference occurred, so nothing about this model's
throughput has been measured.** Full triage in [`docs/M0_TRIAGE.md`](docs/M0_TRIAGE.md).

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

> Deferring things is expected and fine. Deferring them *silently* is what costs points.

### 3.2 Component contracts

> §7.2 and §7.4 both grade this. Show the interface that makes the ML model one bounded, swappable piece — and prove it by shipping a second trivial implementation (a static-image stub is enough).

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

> State how you measured, not just the number. A timestamp at ingress and a timestamp at browser paint are very different measurements from two `time.time()` calls around a function.

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

> Be specific: "an L4 instead of a T4 gets us from X to Y fps" beats "more GPU."

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

> Test from §8: could another senior engineer execute this without you in the room? Write it as a runbook, not a narrative.

### 5.1 Preconditions

> What must be true before phase 1 starts — quality bar met, observability deployed on the vendor path first so you have a baseline to compare against.

### 5.2 Phase 1 — Shadow mode

| | |
|---|---|
| Traffic served by | Vendor |
| In-house path | Renders in parallel, output discarded |
| Duration | |
| Compared | Latency distributions, failure rates, sync quality |
| Exit criteria | |
| Cost | Double render cost — state it |

### 5.3 Phase 2 — Flagged rollout

| Stage | Cohort | Duration | Promote if | Abort if |
|---|---|---|---|---|
| 1 | Internal only | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | 100% | | | |

> Cohorts should be chosen so a bad outcome is survivable — internal interviews before real candidates. Say why you chose the order you chose.

### 5.4 Rollback

> Config flip, not a deploy. Target time-to-rollback. Who can execute it without approval. What happens to sessions in flight at the moment of rollback — this is the detail that separates a plan from a hope.

### 5.5 Decommission

> When the vendor contract is actually cancelled, and how long you keep the integration warm as an escape hatch after full cutover.

---

## 6. Sources

> Numbered, so §1 tags can reference them. Separate primary sources from secondary ones — a vendor's own docs and a published paper are primary; a listicle blog post is not, and citing one as though it were undermines the whole confirmed/inferred framing.

**Primary**

1.

**Secondary / context**

1.
