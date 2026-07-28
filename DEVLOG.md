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
