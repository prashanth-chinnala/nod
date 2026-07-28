# PROCESS.md

Real-time conversational avatar — architecture research, prototype, and build-vs-buy recommendation.

| | |
|---|---|
| Author | *your name* |
| Time spent | *actual hours, honestly* |
| Hardware used | *e.g. Colab T4 16GB / RTX 3090 / CPU-only M2* |
| Prototype status | *what works, what doesn't* |

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
| End-of-turn detection | 100–300 | NOT YET MEASURED | | Often the largest and least-discussed term |
| Speech-to-text finalize | 50–150 | NOT YET MEASURED | | Streaming, so mostly already done incrementally |
| LLM time-to-first-token | 200–500 | NOT YET MEASURED | | Only TTFT matters, not total generation |
| TTS time-to-first-audio | 100–300 | NOT YET MEASURED | | Sentence-chunked, not whole-response |
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
| Setup fragility | | §6 requires a clean-clone build |

### 2.2 Candidates evaluated

| Model | fps / latency | License (code / weights) | Streaming? | Verdict |
|---|---|---|---|---|
| | | | | |

> Include at least one model you rejected **on license** and one you rejected **on latency**. That demonstrates the criteria were real rather than decorative.

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
| Audio in → video out | Deferred to M2 | Blocked on M0 model spike; needs GPU |
| Browser streaming transport | Deferred to M3 | |
| Session start/stop lifecycle | Built (M1) | State machine complete and tested against the stub renderer |
| Idle-loop fallback frame | Built (M1) | `IdleLoop` + `FrameMixer`; real frames land in M4 |
| Interruption handling | Built (M1) | Turn-epoch cancellation, tested |
| Warm model pooling | Deferred (M7) | Out of prototype scope; described in §1.4 |
| Multi-session concurrency | Deferred (M7) | One orchestrator per session is wired, but only one session is exercised |
| STT / LLM / TTS integration | Deferred to M4 | Stubbed behind async generators so the state machine is testable without them |

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

| Metric | Measured | Hardware | Method |
|---|---|---|---|
| Audio-in to first-frame-out | NOT YET MEASURED | | |
| Steady-state fps | NOT YET MEASURED | | |
| Output resolution | NOT YET MEASURED | | |
| Interruption → avatar silent | NOT YET MEASURED | | |
| Peak VRAM | NOT YET MEASURED | | |

> State how you measured, not just the number. A timestamp at ingress and a timestamp at browser paint are very different measurements from two `time.time()` calls around a function.

### 3.4 Gap to production real-time

| Gap | Cause | What closes it | Est. cost |
|---|---|---|---|
| | | | |

> Be specific: "an L4 instead of a T4 gets us from X to Y fps" beats "more GPU."

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
