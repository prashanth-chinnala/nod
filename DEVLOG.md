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

---

## Session 3 — M0 handed off, M4's turn-taking built

**Attempted:** make M0 runnable by someone who has not used Colab, then build the half
of M4 that is not blocked on hardware or API keys.

**Worked.** 172 tests in 0.16s, ruff + format + mypy strict clean, and 17/17 smoke
assertions over a real socket — including a new one that drives a whole turn with
synthetic microphone audio and no button presses.

### M0 handoff

`docs/M0_SPIKE.md` is the runbook; `docs/M0_FOR_BEGINNERS.md` is the same job written
for someone new to Colab, with a glossary and the two facts about Colab that bite you
if nobody tells you (the runtime is wiped when you close the tab, and the free tier
does not guarantee a GPU). `notebooks/m0_musetalk_spike.ipynb` is the timed harness.

### M4, the part that could be finished

Split into policy and detector, on purpose:

| File | What | Verified? |
|---|---|---|
| `audio/turn_detection.py` | Onset, hysteresis, retraction, end-of-turn as separate decisions | Yes — 30 tests over probability sequences |
| `audio/vad.py` — `EnergyVad` | RMS gate, zero dependencies | Yes |
| `audio/vad.py` — `SileroVad` | Real VAD via torch.hub | **No. Never executed.** |
| `server.py` | Mic bytes → VAD → policy → orchestrator | Yes, end-to-end |
| `web/index.html` | Streams mic PCM; server decides turns | Yes, by the smoke script's synthetic client |

The split is the point. Silero answers "is this 32ms speech?" and nothing else. Every
decision that matters — how certain before interrupting, how long a pause is still the
same turn, when a false onset should be retracted — is in `turn_detection.py`, which
imports nothing and is testable as a table of floats. Thresholds are otherwise tuned by
ear against a recording nobody else has.

Two things fell out of writing the tests rather than being designed in:

- **Inverted hysteresis is now rejected at construction.** A release bar above the onset
  bar makes speech harder to sustain than to enter, which produces an end-of-turn inside
  almost every word. That presents as a wildly over-eager avatar and is very hard to
  trace back to two numbers in a config, so it fails loudly instead.
- **The onset run must be unbroken.** Counting non-consecutive loud frames toward onset
  would make the detector steadily more trigger-happy the noisier the room got.

One test of mine failed legitimately and the code was right: I fed 192ms of speech into
a detector with a 200ms floor and expected an end-of-turn. It correctly retracted.

### `SileroVad` is unverified — stated plainly

There is no torch in this development environment, so `SileroVad` has been written,
wired, type-checked, and **never run**. It is excluded from CI, checked only
structurally in the suite, and flagged in its own docstring. Treat its first execution
as work, not a formality.

Writing it anyway was deliberate: the whole claim of M4's design is that the detector is
swappable behind a Protocol, and `AVATAR_VAD=silero` is that claim made concrete. The
verified path — policy, server wiring, client, tests — runs on `EnergyVad`, which needs
nothing and works on a clean clone.

### The measurement worth arguing about

`turn_detect = 700ms`. It is not a measurement; it is the configured silence window, and
it appears in the latency budget unchanged no matter what hardware runs underneath.

That makes it the most interesting number in the budget so far: if the target is a
sub-second perceived turnaround, **this single configuration value is over two thirds of
it**, and it is the one term a faster GPU cannot touch. Recorded as caveat 4 in
`PROCESS.md` §3.3.1 rather than quietly presented as an achievement.

### Deferred, and why

- **Real STT, LLM, TTS.** All need API keys, which are not present. The adapters are
  cheap once there is something to authenticate against; writing three unverified
  network clients is not.
- **`scripts/prepare_idle_loop.py`.** Needs ffmpeg (not installed) and a real reference
  clip (does not exist until M0/M2). Writing a script whose inputs do not exist and
  which cannot be run would be the same mistake as `SileroVad`, twice.
- **Mic capture uses `ScriptProcessorNode`**, deprecated in favour of `AudioWorklet`.
  The worklet needs a separate module file or a Blob URL, and the README promises a
  clean clone with no build step. Documented shortcut, same class as the WebSocket
  standing in for WebRTC.

### Still `[HUMAN]`

Unchanged. Nothing has been written into a judgment section.

**Next:** M0 on Colab, which is with the candidate. Then M2. The LLM and TTS adapters
land whenever keys appear — neither blocks anything else.

---

## Session 4 — Claude interviewer, and M0 run 1 failed in setup

### The abort leak (the real find of this session)

Adding a metered LLM exposed a bug that a local placeholder had made invisible. A
barge-in abandoned a turn logically — the epoch check stops the orchestrator reading —
while leaving the provider generating. **Three separate wrappers each failed to
propagate the close**, and any one of them defeats the guarantee alone:

1. `_run_turn` returned out of its `async for` without closing the stream
2. `chunk_into_sentences` drained the token generator without closing it, so a close
   that reached the chunker stopped there
3. the adapter drained the SDK's token generator without closing it

Root cause is the same in all three: **`async for` does not close the iterator it
drains**, and Python defers generator finalization to the garbage collector. With
`ToneTTS` this cost nothing — abandoning a local generator is free. With a metered API
it is a billing leak on every interruption, which is the single action this system is
designed to make cheap.

All three now use `contextlib.aclosing`, and the contract states the requirement rather
than relying on comments: `SentenceStream` and `SpeechStream` return `AsyncGenerator`,
not `AsyncIterator`, because closeability is part of the interface. `mypy` caught the
gap the moment `aclosing` was introduced, which is what prompted tightening the
Protocol instead of casting at the call site.

Worth noting what did *not* find this: 172 passing tests, all green. The bug was
invisible to every one of them because the fake TTS and fake LLM cost nothing to
abandon. It took wiring a real metered backend.

### M0 run 1 — failed, and correctly reported as failed

Tesla T4 15360 MiB on first attempt, so the hardware question is settled. Inference
never ran: `peak_vram_mib: 3`, `weights_on_disk: 96M` against checkpoints totalling
several GB, `exit_code: 1` on all four runs, no output video.

The harness worked exactly as intended. It flagged `"no output video — nothing has been
proven yet; do not record any fps number"` and it is the reason no fabricated figure
entered the write-up. The timing fields it produced — `cold_warm_ratio: 2.1`,
`identity_prep_s: 0.25` — are the ratio between two crashes and the difference between
two identical failures respectively, and `PROCESS.md` §2.2.1 says so explicitly.

**The genuine finding is a fragility one, and it is worth more than the throughput
number would have been:** `download_weights.sh` exited **0** having fetched 96MB, and
`pip install` exited **0** in **13 seconds** for a project depending on `mmcv` and
`mmpose`. Both reported success without doing their job. §2.1's "setup fragility"
criterion now has concrete evidence in it.

Triage cell written to `docs/M0_TRIAGE.md` — it discriminates between the two suspects
(LFS pointer files vs missing OpenMMLab packages) and captures the stderr the JSON
block did not carry.

### Still `[HUMAN]`

The three §2.3 questions were put to me directly this session and I did not answer them.
They are the graded honesty claim about the candidate's own reasoning, and answering
them would defeat the point of asking. Raw material assembled; conclusions not written.

**Next:** the triage cell, then Deepgram TTS/STT (key verified: Aura-2 returned 79KB of
16kHz PCM). Anthropic completions blocked on account credit, not on code.

---

## Session 5 — real voice, and the measurement that reframes the budget

**Attempted:** replace the placeholder tone with Deepgram Aura, and add an OpenAI
adapter alongside the Anthropic one.

**Worked.** 199 tests. 17/17 end-to-end with real synthesised speech through the whole
pipeline — VAD, turn policy, LLM, TTS, renderer, transport, client acks.

### Two format facts that came from measuring, not reading

1. **`/v1/speak` returns `audio/wav` by default.** The 44-byte RIFF header would have
   been handed to the browser as PCM — an audible click at the start of every sentence —
   and would have skewed the byte-to-duration arithmetic the renderer uses to drive the
   mouth. `container=none` returns bare `audio/l16`. Confirmed by inspecting the first 16
   bytes of both responses; asserted in the suite so it cannot be dropped.
2. **Cold TTFB is ~3x warm** (~1020ms vs ~380ms). That is why the HTTP client is
   constructed once per process — a client per request would put every turn on the cold
   path.

### The A/B that matters

Same pipeline, one component swapped, both runs passing all 17 assertions:

| Stage | `ToneTTS` | Aura-2 | Delta |
|---|---|---|---|
| tts_first_audio | 124ms | **893ms** | +769ms |
| avatar_first_frame | 404ms | **1226ms** | +822ms |
| perceived_total | 430ms | **1235ms** | +805ms |

**Real TTS alone exceeds the entire sub-second target.** With the 700ms silence window, a
full turn is ~1.9s from the candidate stopping to the avatar starting — roughly twice
target — and *neither* dominant term touches a GPU.

That reframes §3.4 substantially. Before this measurement the implicit assumption was
that the renderer would be the bottleneck and more GPU would be the answer. It is not:
even a zero-latency model leaves ~1.9s. The real levers are a streaming TTS interface
instead of request/response, and the silence-window policy choice.

Worth noting the placeholder was not merely inaccurate here — it was *optimistic in a
way that hid the actual problem*. 124ms of fake TTS made the budget look comfortable.

### OpenAI adapter, and Ollama for free

Added alongside the Anthropic one rather than replacing it: two unrelated providers
behind one `SentenceStream` is stronger evidence for the swappable-boundary claim than
either alone, and the module docstring tabulates what the boundary actually absorbs
(system-prompt placement, `max_tokens` vs `max_completion_tokens`, sampling accepted vs
rejected, thinking knob vs none, two different stream shapes).

The useful consequence: Ollama, LM Studio, and vLLM all speak the OpenAI wire format, so
a free local model needs **no new adapter** — only `OPENAI_BASE_URL` and a model name. A
local endpoint authenticates nothing, so a placeholder key is supplied rather than
demanded.

### Both LLM providers are billing-blocked, not code-blocked

Anthropic: `400 invalid_request_error`, credit balance too low. OpenAI: `429
insufficient_quota`. Both keys authenticate. Swapping vendors does not help; the blocker
is account credit on both. Recorded in both module docstrings as structurally complete
and never verified against a live completion.

**Next:** Deepgram streaming STT (turns still carry `[Nms of speech, no transcriber]`),
and Aura's WebSocket interface, which §3.3.2 now identifies as the single largest
available latency win.

---

## Session 6 — the LLM unblocked, repo published, docs brought current

### The LLM was never a code problem

Three providers, three billing walls, and swapping vendors did not help:

| | |
|---|---|
| Anthropic | `400 invalid_request_error` — credit balance too low |
| OpenAI | `429 insufficient_quota` |
| Ollama Cloud | `403` on 11 of 19 models — subscription required |

All three keys authenticated. The lesson worth keeping: an authenticating key with a
paywalled account produces a *different* error class from a bad key, and reading which one
you have saves a lot of time. A `401` is a credential problem; a `400`/`429`/`403` means
the request arrived and something downstream said no.

**What unblocked it was reading the whole list instead of the first three entries.** My
initial probe picked `deepseek-v4-flash`, `glm-5.1`, and `kimi-k2.6` — all premium, all
403. Enumerating every model showed **8 of 19 are accessible** on that key. `gpt-oss:20b`
measured **1914ms median time-to-first-sentence**, the fastest available, and its questions
genuinely follow up on what the candidate said. I had told the operator the key was not
usable; that was wrong, and it was wrong because of sampling rather than anything about
the key.

### `.env` did nothing

Every knob in this project is an environment variable, and nothing read the file. Each run
needed `set -a && . ./.env && set +a` in front of it, and forgetting produced a session
that silently fell back to every placeholder — no error, just quietly the wrong system.
That is worse than having no config file at all.

`avatar/config.py` now loads it at server import: twenty dependency-free lines, a real
environment variable always wins, a missing file is not an error, and the parser handles
exactly the `KEY=value` subset this project writes rather than half-implementing shell
quoting. `GET /config` reports which implementation each boundary resolved to and which
variable *names* came from the file — names only, since most of them are credentials.

### Published

23 commits pushed to <https://github.com/prashanth-chinnala/nod>. Before pushing, every
blob in every commit and every commit message was scanned for all four key patterns —
clean. `.env` is gitignored and untracked. (Session 7 removed the `.env.example` exemption
that this session added; see below.)

Worth doing that scan rather than assuming: this repo has to be public, and a key committed
to a public repo is scraped in minutes and cannot be un-published.

### Documentation debt paid

The README documented **none** of the environment variables. The only way to discover that
a real voice was one variable away was to read `server.py` — a clean-clone failure for a
graded artifact. It now carries the full variable table, the `.env` setup, and the
OpenAI-compatible-endpoint trick that was buried in a docstring.

Also added `docs/DEMO_SCRIPT.md`: six manual tests, each naming what to watch **and what it
does not prove**. That last column is the point — it is easy to record a demo that implies
a working talking-head model because a mouth moves in time with speech. The mouth moving
proves the audio-to-video mechanism, not a face.

### M0 diagnosed properly

Read MuseTalk's actual README and `download_weights.sh` instead of guessing. Three causes,
and the first was mine:

1. **My inference command was incomplete.** v1.5 needs `--version v15`,
   `--unet_model_path`, `--unet_config`, `--ffmpeg_path`. Without them it loads the v1.0
   checkpoint layout, which is exit code 1 on its own. My own cell told the operator to
   read the README for the invocation and I had put a guess in the next cell.
2. **The downloader is built to fail silently.** It points `HF_ENDPOINT` at a mirror, has
   no `set -e`, and validates nothing — so 96MB with exit code 0 is the expected shape of
   total failure. 96MB is about `resnet18` plus configs: only the `curl` step worked.
3. **Python 3.12 against a 3.10-pinned stack.** `mmcv==2.0.1` has no 3.12 wheel, which
   explains the other implausible number — a 13-second `pip install` for a stack that takes
   minutes. It never installed OpenMMLab.

`notebooks/m0_musetalk_v2.ipynb` fixes all three, gates on imports resolving *before*
downloading gigabytes, audits every checkpoint by size (detecting git-lfs pointers and
Google Drive quota HTML by their first bytes), and computes `inference_actually_ran` from
exit code **and** an output file **and** VRAM passing 500 MiB — so the block that gets
pasted back cannot be mistaken for a measurement when it is a failure.

### Hosting path, de-risked before the model

`notebooks/run_on_colab.ipynb` puts the server on a Colab GPU behind a free cloudflared
HTTPS tunnel, so a locally-opened page streams voice up and video down. Deliberately
runnable **today with the stub**, because it proves four things unrelated to the model that
would each sink the demo alone: that the tunnel proxies WebSockets, that the page gets a
secure context (browsers refuse microphone access without one), that the client upgrades to
`wss://`, and that the runtime survives. Learning that a tunnel breaks WebSockets after two
days on the model would be an expensive ordering.

### Still `[HUMAN]`

Unchanged. The three §2.3 questions were put to me directly again and I did not answer them.

**Next:** M0 run 2, then M2. Aura's WebSocket TTS is measured at **351ms flat** against
907ms over REST and remains the largest unbuilt latency win.

## Session 7 — env files layered, and the template deleted

### `.env.development`, and why it is three files rather than a rename

The obvious move was `mv .env .env.development` and a one-word change in the loader. What
went in instead reads three candidates in descending precedence — `.env.development`,
`.env.local`, `.env` — because a bare rename gives up the thing the name implies. Naming a
file after an environment is only useful if a *different* environment can have its own,
and that only works if shared settings can stay in one file while the handful that differ
live in another. A rename would have forced every value to be duplicated per environment.

`load_env` only fills variables that are *unset*, so precedence falls out of load order for
free. Four rules, each with a test that names what it prevents:

- **A real environment variable beats every file.** `AVATAR_TTS=tone uvicorn ...` still
  overrides, and CI cannot be clobbered by a stray file.
- **The search stops at the first directory containing any candidate.** Without the stop, a
  `.env` two levels up layers under the repo's own `.env.development` and contributes a
  value that cannot be accounted for from inside the project — the phantom-setting bug.
- **`AVATAR_ENV_FILE`** names one file and skips the search, for a mounted secret or a path
  outside the repo. A caller who names a file does not want a merge with its neighbour.
- **An unreadable file degrades to defaults** rather than refusing to boot.

`GET /config` now reports `env_files_read`. That field exists for one specific failure:
`/config` showing `scripted` when you expected `openai` is either a wrong value or a file
that was never opened, and those have completely different fixes. Names only, never
contents — most of what is in there is a credential.

`tests/test_config.py` covers all of it, 12 tests. Precedence deserves tests more than
parsing does: a mis-parsed line is a visible error, while wrong precedence presents as *"the
setting had no effect"* with nothing to grep for.

### `.env.example` deleted

It was the one tracked env file, kept as a fill-in-the-blanks template. Removed at the
user's request, and the reasoning holds up: a tracked `.env*` path is one `git add -f`
away from a tracked key, and the exemption line in `.gitignore` was the only thing standing
between the two. `.gitignore` now covers `.env` and `.env.*` with **no exemption**, so the
verification is a single command with no special case to remember:

```bash
git ls-files | grep -E '^\.env'      # must print nothing
```

The template did do a real job, though — it was the only place a clean clone learned the
variable *names*. That job moved to README's Configuration table, which already listed
every variable with its default and options, so nothing was lost but the trap.

### Still `[HUMAN]`

Unchanged. The three §2.3 questions and all of §4 remain unanswered by me.

**Next:** unmoved — M0 run 2, then M2. Aura's WebSocket TTS is still the largest unbuilt
latency win at **351ms flat** against 907ms over REST.
