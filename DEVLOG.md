# DEVLOG.md

Raw material for `PROCESS.md` §3.1. Written as it happens, because reconstructing it
on day 12 produces a much weaker document.

---

## Session 1 — repo bootstrap, M1 complete

**Attempted:** land the handoff docs as repo artifacts, then build the M1 spine.

**Worked.** `pytest -m "not gpu"` → 89 passed in 0.08s. `ruff check`, `ruff format
--check`, and `mypy src/avatar` all clean. That is the full CI-equivalent chain, so
M1's acceptance criterion is met locally; it has not yet run on GitHub Actions
because there is no remote.

Shipped:

| File | What |
|---|---|
| `src/avatar/contracts.py` | `Frame`, `AudioChunk`, `Turn`, `Message`, and four Protocols |
| `src/avatar/state.py` | `State`, `LEGAL_TRANSITIONS`, `FRAME_SOURCE` — tables, not if-chains |
| `src/avatar/telemetry.py` | `Telemetry` with named call sites + in-process histograms |
| `src/avatar/mixer.py` | `IdleLoop`, `FrameMixer` — constant cadence, pts ownership |
| `src/avatar/orchestrator.py` | `SessionOrchestrator`, `heard_text` |
| `src/avatar/renderers/` | `build()` registry + `StubRenderer` (BMP, no deps) |
| `tests/` | 5 files, 89 tests, fake clock throughout |
| `.github/workflows/ci.yml` | lint / format / types / tests, no GPU |
| `web/mockup.html` | the design mockup, cleaned of encoding damage |

### Audit of the handed-over `orchestrator.py` sketch

The sketch was structurally right — epoch cancellation, continuous track, renderer
behind a Protocol, history truncated to what was heard. Four defects found while
porting it, all fixed with tests that pin them:

1. **The silence re-prompt could never fire.** `on_idle_tick` guarded on
   `state == IDLE`, then delegated to `on_end_of_turn`, which returns early unless
   `state == LISTENING`. Mutually exclusive, so the branch was dead. The re-prompt
   now starts the turn directly, and `IDLE → THINKING` is an explicit entry in the
   transition table. Pinned by `test_idle_reprompt_fires_after_the_timeout`.

2. **A barge-in could be lost between `create_task` and the task's first tick.**
   The epoch was incremented inside `_run_turn`'s body. In the window before the
   task first ran, a cancellation incremented the epoch, and then the task
   incremented past it and generated a turn that had already been abandoned. The
   epoch is now bumped synchronously in `_begin_turn` and passed in as an argument.
   Pinned by `test_barge_in_during_thinking_does_not_wedge`.

3. **Frame-source selection was scattered.** The sketch's docstring said to keep it
   in one place; `_transition` handled only the idle direction and the pipeline
   called `mixer.select_renderer()` from the middle of the TTS loop. Source is now a
   pure function of state via `FRAME_SOURCE`, applied only in `_transition`. Pinned
   by `test_every_state_has_transitions_and_a_frame_source`.

4. **History truncation keyed on the wrong quantity.** `spoken_ms` accumulated
   `chunk.duration_ms` as chunks were handed to the transport, and its own comment
   said "audio confirmed flushed to transport" — but §1.6 of the brief asks for
   "audio frames actually acknowledged as played, not frames sent." The gap is the
   client's jitter buffer, which a barge-in discards. `Turn` now tracks
   `audio_sent_ms` and `audio_played_ms` separately, and `on_audio_played` is the
   only input that moves the latter. Pinned by
   `test_sent_but_unplayed_audio_is_not_credited`.

Also: `IdleLoop.at_clean_exit()` was defined and never called, so the seam
constraint it exists to enforce was not enforced. The handover now waits for a
mouth-closed frame, bounded by `SEAM_WAIT_MAX_MS` (120ms) so a sparsely-annotated
clip cannot delay speech indefinitely; forced handovers increment `seam_forced`.

Two smaller reconciliations: the `TalkingHeadRenderer` signatures in `PROCESS.md`
§3.2 and in the sketch disagreed (`push_audio(pcm: bytes)` vs `push_audio(chunk:
AudioChunk)`, `cancel()` vs `reset()`). Took the sketch's version — the chunk needs
to carry its epoch, which is how the renderer tags the frames it produces without
knowing what an epoch means. `PROCESS.md` §3.2 updated to match.

### Deliberate deviations from the guide

- **`web/mockup.html`, not `web/index.html`.** The handed-over file is a design
  mockup driven by `setInterval`; naming it `index.html` would mean M3 either
  overwrites it or ships simulated numbers. It keeps its own name and M3 writes the
  real client alongside it.
- **`Turn.heard_text()` became `orchestrator.heard_text(turn)`.** The guide puts
  `Turn` in `contracts.py` and also says that file holds no logic. Truncation is
  policy, so it moved to the module that owns the policy; `Turn` stayed a dataclass.
- **`tests/test_boundaries.py` is a fifth test file.** The guide names four and
  separately requires the import boundary be enforced in CI. It parses the AST of
  each orchestration module rather than importing and catching ImportError, because
  the claim is that the dependency is absent from the graph — not that it happened
  not to be installed on the machine running the suite.
- **`LLM`/`TTS`/`Transport` are injected Protocols, not module-level stubs.** The
  sketch had `llm_stream_sentences` and `tts_stream` as module functions raising
  `NotImplementedError`, which cannot be substituted in a test. They are now
  constructor arguments, which is what makes the 89 tests possible at all.

### Deferred, and why

- **M0 (model spike) — blocked on hardware.** This is the highest-risk unknown in
  the whole assessment and it cannot be resolved from here: it needs a GPU, and no
  measurement may be invented (Rule 1). M2 cannot start until it resolves. Building
  M1 first inverts the guide's ordering; the justification is that M1 is entirely
  GPU-free and unblocked, and leaving it idle while M0 waits on hardware would have
  produced nothing.
- **`scripts/`, `server.py`, `audio/`, `transport/`, `llm.py`** — M3/M4/M5. The
  Protocols they will implement exist; the implementations do not.
- **`README.md` is still a stub.** It is a graded artifact (clean-clone setup) and
  writing setup instructions before there is anything to set up would mean writing
  instructions nobody has followed. M6 verifies it against a fresh clone.

### `[HUMAN]` gaps outstanding

Nothing has been written into a judgment section. `PROCESS.md` carries `[HUMAN]`
markers on §0 (claim tagging), §2.3 (model selection rationale), and all of §4
(build-vs-buy), and `NOT YET MEASURED` in every cell of §1.5 and §3.3.

**Next:** M0, once hardware is known. Everything downstream of the model choice is
blocked on it.

---

## Session 2 — hardware confirmed, M3 complete

**Hardware:** Colab / Kaggle free tier, T4 16GB. Recorded here because it changes the
model shortlist: Ditto's TensorRT 8.6.1 requirement with GPU-specific prebuilt engines
fights an ephemeral runtime badly, which pushes MuseTalk ahead on setup risk. **The
decision itself is `[HUMAN]`** and has not been made.

**Attempted:** M3 — streaming to a browser. M0 still needs the GPU, and M3 is entirely
GPU-free, so it went first for the same reason M1 did.

**Worked.** 131 tests in 0.12s, ruff clean, ruff format clean, mypy strict clean. Then
the part the suite cannot prove — `scripts/smoke_session.py` drives a real session over
a real socket and checks 15 properties. All 15 pass:

```
states      IDLE -> LISTENING -> THINKING -> SPEAKING -> CANCELLING -> LISTENING
            -> THINKING -> SPEAKING -> CANCELLING -> LISTENING
video       117 frames, 15 before any turn      (25.4fps over 4.6s)
audio       94 chunks, 7520ms
stale drops {'audio': 2}
mixer       0 repeated, 127 discarded
flushes     2
latency     avatar_first_frame=396ms  perceived_total=398ms  llm_ttft=181ms
            tts_first_audio=122ms  interrupt_to_silent=0.6ms
```

Numbers and their caveats are in `PROCESS.md` §3.3.1. The short version: they measure
the session layer and the instrumentation, not any model. `llm_ttft` and
`tts_first_audio` are the stubs reporting their own configured delays back.

Shipped:

| File | What |
|---|---|
| `transport/websocket.py` | 13-byte header codec + `Transport`. Takes two send callables, so it needs no web framework and CI needs no web stack |
| `server.py` | FastAPI, one orchestrator per socket, four background pumps |
| `llm.py` | `chunk_into_sentences` (survives M4 unchanged) + `ScriptedInterviewer` |
| `audio/tts.py` | `ToneTTS` — real duration, real chunking, real pacing, fake voice |
| `idle.py` | Synthetic placeholder loop + the M4 loader for a real prepared clip |
| `bmp.py` | Twenty-line BMP encoder so nothing needs Pillow |
| `web/index.html` | The real client. Canvas, Web Audio, mic, live readouts |
| `scripts/smoke_session.py` | Headless end-to-end verification |
| 42 new tests | Codec round-trips, chunker flush behaviour, TTS timing, idle-loop loading |

### One design change made mid-session, and why

The first run reported `perceived_total` equal to `avatar_first_frame` — 398 vs 396ms.
They were the same instant emitted under two names, because both were recorded when the
server handed the first frame to the mixer. Reporting that as an end-to-end number would
have silently dropped encode, socket, decode, and paint from the budget: exactly the
distinction `PROCESS.md` §3.3 warns about between "a timestamp at ingress and a timestamp
at browser paint."

Fixed properly rather than relabelled. `Turn` gained `first_paint_at`, the orchestrator
gained `on_first_paint`, and the client reports after `drawImage` returns. The server no
longer emits a perceived total at all — it cannot know one.

The 2ms delta above is still not a real browser figure: the smoke script reports paint on
receipt without decoding anything, so it is a lower bound. Noted as caveat 2 in §3.3.1.

### Known gap found by measuring, not fixed

127 frames were discarded on a single barge-in. That is not a bug — they were correctly
invalidated — but it revealed that the mixer's queue is unbounded and grows because
`ToneTTS` generates 4× faster than playback. A six-second utterance queues ~150 frames;
at 256×144 BMP that is ~16MB of buffered video per turn.

The right fix is backpressure keyed on `audio_sent_ms - audio_played_ms`, which the
`Turn` already tracks. Not done, because it needs a timeout path for clients that never
acknowledge, and a half-implemented version that can stall a turn forever is worse than a
documented gap. Recorded in `PROCESS.md` §3.4 with a cost estimate.

### Deliberate deviations, session 2

- **`scripts/smoke_session.py` is not in the guide's file list.** It is the only thing
  that verifies the wire format matches what the client parses. Not in CI, because that
  would mean a web stack in the CI dependency set to test a layer already covered.
- **`bmp.py` and `idle.py` are not in the guide's layout.** Both fell out of needing
  frames a browser can display without an image library. `PROCESS.md` §3.1 records the
  BMP cost as an M2 gap.
- **`web/index.html` uses system fonts, not Google Fonts.** The mockup loads webfonts;
  the real client must not, since the README promises a clean clone with no network
  dependency.

### Still `[HUMAN]`

Unchanged from session 1, plus the model pick now that hardware is known. Nothing has
been written into a judgment section. `PROCESS.md` §3.3 headline rows still read
`NOT YET MEASURED` because the model does not exist.

**Next:** M0 on Colab. If the GPU is unavailable, M4 (VAD, real STT/LLM/TTS) is not
blocked on it and can go first — the Protocols are already in place.
