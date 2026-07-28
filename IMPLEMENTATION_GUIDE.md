# IMPLEMENTATION_GUIDE.md

Handoff brief for continuing development of the Exterview Head of Engineering take-home.
Written to be read by a coding agent (Claude Code or similar) at the start of a session.

> **Suggested use:** drop this at the repo root and reference it in `CLAUDE.md` with
> "Read IMPLEMENTATION_GUIDE.md before starting any task." Re-point the agent at the
> relevant milestone section at the start of each session rather than asking it to
> hold the whole plan in context.

---

## 0. Read this first — three rules that override everything else

**Rule 1 — Never invent a measurement.** This submission is graded on honesty about
constraints. Every latency figure, fps number, VRAM number, and cost estimate must
come from an actual run on actual hardware, or be explicitly labelled as an estimate
with the reasoning shown. If asked to fill in a results table and no measurement
exists, write `NOT YET MEASURED` and say so. A fabricated plausible number is the
single worst possible failure mode here — worse than an empty table, worse than a
bad number. This applies to filling in `PROCESS.md` tables too.

**Rule 2 — Judgment sections belong to the human.** Do not write prose for these:
the build-vs-buy recommendation, the confirmed-vs-inferred tagging in the
architecture document, or the "what would change my mind" thresholds. You may
research, assemble evidence, draft structure, and check internal consistency. The
conclusions are the candidate's, and a grader is specifically testing whether they
are the candidate's. Flag them as `[HUMAN]` and move on.

**Rule 3 — Prefer boring and finished over clever and partial.** The grading rubric
explicitly states that a well-chosen existing open-source model beats a from-scratch
half-working novel approach, and "Clarity over Cleverness." If a task can be done
two ways, take the one that will still be working on submission day.

---

## 1. What is being built, and what it is for

A prototype real-time conversational avatar: audio in, lip-synced talking-head video
out, streamed to a browser, with working session lifecycle and interruption handling.

It is a **take-home assessment for a Head of Engineering role**, not a product. This
changes priorities in ways that are counterintuitive for normal development:

| Normally high priority | Here |
|---|---|
| Visual output quality | Explicitly out of scope. Blurry is fine. |
| Model accuracy / fidelity | Out of scope. Do not fine-tune anything. |
| Feature completeness | Out of scope. Deferring is fine *if documented*. |
| Multi-language support | Out of scope. |
| Security / compliance review | Out of scope. |

| Normally lower priority | Here — highest value |
|---|---|
| Interface boundaries between components | Graded directly |
| Session state machine correctness | Graded directly |
| Tests that run without a GPU | Enables the required CI |
| Documenting what was *not* built and why | Graded directly |
| Instrumentation points | Graded directly |
| Incremental commit history | Graded (single squashed commit is a flag) |

When trading off, trade toward the second table.

---

## 2. Decisions already made — do not re-litigate

| Decision | Value | Rationale |
|---|---|---|
| Frame transport, primary | WebSocket, chunked frames | Documented shortcut the brief explicitly permits |
| Frame transport, stretch | WebRTC via aiortc | Only after milestone M5 is green |
| Video track lifetime | Opens once at session start, closes at session end | Renegotiating per turn causes visible flashes |
| Frame cadence | Constant 25fps from a mixer, always emitting | A stalled track is more visible than a dropped frame |
| Cancellation mechanism | Monotonic turn epoch, stale artifacts dropped at consumer | Task-kill races; an integer write does not |
| Conversation history | Truncated to audio actually played, not generated | Otherwise the LLM's context diverges from reality |
| "Thinking" visual | None — idle loop covers `LISTENING` and `THINKING` | Pure scope; document the choice |
| Renderer coupling | Behind a `TalkingHeadRenderer` Protocol, always | Graded requirement; also enables GPU-free tests |
| Language / runtime | Python 3.11+, asyncio | Every candidate model ships Python inference |
| Server | FastAPI + uvicorn | Boring, has WebSocket support built in |

**Model choice is NOT yet made.** See M0. Two viable candidates:

- **MuseTalk** (Tencent Music) — MIT code, weights permit commercial use, ~30fps+ on
  a V100 at 256x256, has a documented real-time inference mode. Safer setup.
- **Ditto** (Ant Group) — Apache-2.0, explicitly designed for streaming and low
  first-frame delay, ships a streaming pipeline. Better architectural fit for this
  brief, but wants TensorRT 8.6.1 with GPU-specific prebuilt engines, which fights
  the clean-clone requirement.

Do not pick LatentSync (roughly 10x slower than real-time; ~100s for 10s of video on
a 4090) or Wav2Lip (license prohibits commercial use). Both are useful as *documented
rejections* in the model-selection memo.

---

## 3. Repository layout

Create this structure. Do not deviate without saying why in `PROCESS.md`.

```
.
├── README.md                     # clean-clone setup + hardware reqs (graded)
├── PROCESS.md                    # architecture doc, memos, migration plan
├── IMPLEMENTATION_GUIDE.md       # this file
├── CLAUDE.md                     # agent instructions, points at this file
├── pyproject.toml
├── .github/workflows/ci.yml
├── src/avatar/
│   ├── contracts.py              # Protocols + dataclasses. No logic. No imports of impls.
│   ├── state.py                  # State enum + transition table
│   ├── orchestrator.py           # SessionOrchestrator — the state machine
│   ├── mixer.py                  # FrameMixer, IdleLoop
│   ├── telemetry.py              # emit_* hooks; stdout in prototype
│   ├── server.py                 # FastAPI app, session routing
│   ├── renderers/
│   │   ├── base.py               # re-export Protocol
│   │   ├── stub.py               # solid-colour frames, no GPU — REQUIRED
│   │   └── <chosen>.py           # e.g. musetalk.py
│   ├── audio/
│   │   ├── vad.py                # Silero VAD wrapper, turn detection
│   │   └── tts.py                # streaming TTS adapter
│   ├── llm.py                    # sentence-chunked streaming wrapper
│   └── transport/
│       ├── websocket.py
│       └── webrtc.py             # stretch only
├── web/index.html                # single-file client, no build step
├── scripts/
│   ├── prepare_idle_loop.py      # video -> decoded frames + mouth-closed index
│   └── measure_latency.py        # produces the numbers for PROCESS.md §3.3
└── tests/
    ├── test_state_machine.py
    ├── test_epoch_cancellation.py
    ├── test_mixer_cadence.py
    └── test_history_truncation.py
```

Two structural invariants worth enforcing in review:

1. `contracts.py` imports nothing from the rest of the package. Everything else
   imports *from* it. If a circular import appears, a boundary has been violated.
2. Nothing in `orchestrator.py`, `mixer.py`, or `state.py` may import torch, cuda,
   numpy-heavy model code, or any renderer implementation. Test this in CI with an
   import-linter rule or a simple grep test. This is what keeps CI GPU-free, and it
   is the mechanical proof of the "ML model is one bounded, swappable piece" claim.

---

## 4. Milestones

Execute in order. Each has a binary acceptance test. **Do not start a milestone
before the previous one's acceptance criteria pass** — the failure mode for this
assessment is a half-finished everything.

### M0 — Model spike (throwaway, timebox: 1 day)

The single highest-risk unknown is whether the chosen model runs at all on available
hardware. Resolve it before building anything around it.

- In a scratch directory *outside* the repo, get the candidate model to render one
  short clip from one reference image/video plus one audio file.
- Record: wall-clock time, output fps, peak VRAM, resolution, and every setup
  problem hit. This log becomes evidence in the model-selection memo.
- Try the second candidate too if the first takes more than half a day to run.

**Acceptance:** a rendered video file exists, and a written note states which model
was chosen and the measured throughput on the actual hardware.

**If both candidates fail on available hardware:** stop and escalate to the human.
Options are a lighter model, a smaller resolution, or Colab/Kaggle free GPU. Do not
silently spend three days fighting a CUDA install.

### M1 — Skeleton, contracts, stub renderer, CI green

No GPU involved. Build the testable spine first.

- `contracts.py`: `Frame`, `AudioChunk`, `Turn`, `TalkingHeadRenderer` Protocol.
- `state.py`: `State` enum and the transition table as data, not `if` chains.
- `orchestrator.py`, `mixer.py`: port from the existing `orchestrator.py` sketch.
- `renderers/stub.py`: emits solid-colour frames at a configurable simulated
  latency. Must support `reset()` as a real no-op-safe operation.
- All four test files, using the stub renderer only.
- `.github/workflows/ci.yml` — lint (ruff), types (mypy on `src/avatar`), tests
  (pytest). No GPU, no model weights, no network.

**Acceptance:** `pytest` passes locally and in GitHub Actions. `ruff check` clean.
The boundary test proves no torch import in the orchestration modules.

### M2 — Real renderer behind the interface

- `renderers/<chosen>.py` implementing the Protocol. Model-specific preprocessing
  (face detection, parsing, latent encoding of reference frames) goes in
  `prepare_identity`, which is allowed to be slow — it runs once, offline.
- `push_audio` / `frames` must work on chunks, not whole files. If the upstream repo
  only offers a batch `main()`, wrap it in a chunked loop and note the inefficiency
  in `PROCESS.md`; do not restructure their model code.
- Offline path: WAV file in, MP4 out, via the Protocol.

**Acceptance:** the same test suite passes against both `stub` and the real renderer
(mark the real one `@pytest.mark.gpu` and exclude from CI). Swapping renderers is a
one-line config change — demonstrate it.

### M3 — Streaming to a browser

- `transport/websocket.py`: length-prefixed JPEG or WebP frames plus a separate
  audio channel. Include the frame `pts_ms` so the client can detect drift.
- `web/index.html`: canvas for frames, Web Audio for playback, mic capture, and a
  visible state readout (`IDLE` / `LISTENING` / `THINKING` / `SPEAKING`). Plain JS,
  no build step — the README must not require npm.
- Surface `frames_repeated` and measured fps live in the page. This doubles as the
  demo instrument for the Loom video.

**Acceptance:** open the page, play an audio file, see lip-synced video. The visible
state readout changes correctly.

### M4 — Session mechanics (the highest-graded milestone)

- `scripts/prepare_idle_loop.py`: decode a short neutral clip to frames, and emit a
  `mouth_closed_indices` set. Approach: per-frame mouth-openness from face landmarks,
  threshold it. If landmark detection is a rabbit hole, hand-annotate the indices in
  a JSON file and document that as a deliberate shortcut.
- `audio/vad.py`: Silero VAD. Expose speech-onset and end-of-turn as distinct events
  with separately tunable thresholds — they are not the same decision.
- Wire the orchestrator into `server.py`: one orchestrator per session.
- Barge-in must be visible in the browser within a stated target from speech onset.

**Acceptance:** speak over the avatar mid-sentence and it goes quiet and returns to
the idle loop, *without* finishing the stale render. Verify from logs that stale-epoch
frames were dropped rather than that it merely looked right.

### M5 — Measurement and telemetry

- `telemetry.py`: histograms for each stage in the `PROCESS.md` §1.5 table. Print to
  stdout as structured JSON in the prototype; note in the architecture doc where
  these would go in production (OTel spans, one trace ID per conversational turn).
- `scripts/measure_latency.py`: runs N turns, reports p50/p95 for audio-in to
  first-frame-out, steady-state fps, interrupt-to-silence, peak VRAM.
- Run it. Paste the real output into `PROCESS.md` §3.3. See Rule 1.

**Acceptance:** `PROCESS.md` §3.3 and §1.5 contain real measured numbers with the
hardware named, and the measurement method described.

### M6 — Documentation pass

- `README.md`: clean-clone setup, hardware requirements, model weight download steps,
  how to run, how to run tests, what works and what does not. Verify by cloning to a
  fresh directory and following your own instructions literally.
- `PROCESS.md`: fill every non-`[HUMAN]` section. Cross-check that stated targets and
  measured numbers agree, and where they do not, that the gap is explained. The
  rubric grades whether the document and the prototype describe the same reality.

**Acceptance:** a fresh clone reaches a running prototype using only the README.

### M7 — Stretch, only if M0–M6 are all green

In priority order: WebRTC transport via aiortc → warm renderer pool with session
leasing → concurrent session support. Each is optional. Attempting M7 at the cost of
M5 or M6 is a net loss.

---

## 5. Test specifications

These four files are the load-bearing evidence for "Systems depth." Write them
against the stub renderer so they run in CI.

**`test_state_machine.py`**
- every transition in the `state.py` table fires from the correct precondition
- `on_end_of_turn` while in `IDLE` is a no-op, not a crash
- VAD retraction from `LISTENING` returns to `IDLE` without starting a turn
- idle re-prompt fires after the timeout, and only from `IDLE`
- an exception mid-turn returns state to `IDLE` rather than wedging

**`test_epoch_cancellation.py`**
- barge-in during `SPEAKING` → state is `LISTENING`
- frames offered with a pre-barge-in epoch after cancellation are dropped
- barge-in during `THINKING`, before any frame rendered, does not wedge
- two barge-ins in rapid succession yield one clean `LISTENING`
- audio chunks arriving with a stale epoch are not forwarded to transport

**`test_mixer_cadence.py`**
- emits exactly 25 frames per simulated second
- no gap across an idle→renderer source switch
- renderer starvation repeats the last frame and increments `frames_repeated`
- `select_idle()` drains the pending render queue

**`test_history_truncation.py`**
- uninterrupted turn stores the full generated text
- interrupted turn stores only the played prefix, marked interrupted
- interruption before any audio played stores nothing, or an empty marker
- history never contains text the client did not receive

Use a controllable fake clock rather than `asyncio.sleep` in tests — real sleeps make
the suite slow and flaky, and a flaky suite in CI reads worse than no suite.

---

## 6. CI workflow

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: ruff check src tests
      - run: ruff format --check src tests
      - run: mypy src/avatar
      - run: pytest -m "not gpu" --tb=short
```

Keep `[dev]` extras free of torch and any model dependency. If CI needs a GPU
package to import, a boundary has been broken — fix the boundary, do not add the
dependency.

---

## 7. Known traps

| Trap | Symptom | Handling |
|---|---|---|
| Upstream repo has no installable package | `pip install` fails, imports break | Vendor as a git submodule or pinned commit; wrap, don't fork-and-edit |
| Model expects 25fps input video | Lips drift over long clips | Resample the reference clip with ffmpeg during `prepare_identity` |
| First inference call is far slower than steady state | First-frame latency looks terrible | Warm up with a dummy forward pass in `start_session`; report both cold and warm |
| Renderer emits frames in bursts, not steadily | Visible stutter despite adequate average fps | This is what the mixer's lead-in buffer exists for; tune and report the depth |
| VAD fires on the avatar's own audio | Avatar interrupts itself in a loop | Gate VAD on the client mic track only, or use echo cancellation; note which |
| Barge-in "works" but stale audio still plays | Interruption looks laggy | Flush the *client* audio buffer too; server-side flush is not sufficient |
| Idle loop visibly jumps | Obvious cut every few seconds | Pick a clip whose first and last frames match; cross-fade a few frames |
| Model weights are gigabytes | Repo unusable from clean clone | Never commit weights; script the download and document the size in README |

---

## 8. Commit and session discipline

Real incremental commit history is graded. Practically:

- One commit per coherent step, message stating *why* not just what.
- Commit at each milestone boundary at minimum, and generally more often.
- Never squash the history into a single commit before submitting.
- Do not commit: model weights, `.mp4` outputs, `__pycache__`, virtualenvs, API keys.
- Add a `.gitignore` in M1, not later.

At the end of each working session, append a short entry to a `DEVLOG.md`: what was
attempted, what worked, what was deferred and why. This is raw material for
`PROCESS.md` §3.1, and reconstructing it from memory on day 12 produces a much
weaker document than writing it as you go.

---

## 9. What the human must do — do not attempt these

| Item | Why |
|---|---|
| The build-vs-buy recommendation and its thresholds | The entire point of the assessment; must be the candidate's own reasoning |
| Confirmed vs. inferred tagging | An honesty claim about what *they* verified |
| The Loom video | Delivered to camera as a briefing |
| Approving the model choice | A judgment call with a memo attached to it |
| Any number in a results table | Rule 1 |
| The final read of `PROCESS.md` for voice and consistency | It has to sound like one engineer wrote it |

When a task touches one of these, produce the scaffolding, mark the gap `[HUMAN]`,
and report it clearly at the end of the session rather than filling it in.

---

## 10. Definition of done

The submission is complete when all of the following are true:

- [ ] Fresh clone → working prototype using only `README.md`
- [ ] `pytest -m "not gpu"` green in GitHub Actions
- [ ] Audio in → lip-synced video out in a browser
- [ ] Session start/stop works; repeated sessions do not leak GPU memory
- [ ] Idle loop plays with no visible seam when no audio is present
- [ ] Barge-in visibly stops the avatar mid-sentence; stale frames provably dropped
- [ ] Real measured numbers in `PROCESS.md` §1.5 and §3.3, hardware named
- [ ] Every deferred capability listed in §3.1 with a reason
- [ ] All four memos present in `PROCESS.md`, organised, not one wall of text
- [ ] Every claim in the architecture document tagged, with sources listed
- [ ] Incremental commit history, no committed weights or secrets
- [ ] `[HUMAN]` markers all resolved by the candidate

If time runs short, cut from M7 first, then reduce resolution or fps targets, then
cut concurrent sessions. Do not cut tests, measurement, or documentation — those are
where the grading weight sits.
