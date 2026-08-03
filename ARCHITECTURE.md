# Architecture

How `nod` is put together, and why. The organising idea is that **the parts most likely to change
are held behind boundaries that a test enforces**, so that swapping a model, a provider or a
transport is a new file rather than a migration.

That claim is checkable. `tests/test_boundaries.py` fails the build if any of it stops being true.

---

## 1. The shape

Three applications, one runtime.

```
apps/web          Next.js console + the interview room a candidate opens
apps/api          the runtime: state machine, turn taking, renderers, store   (53 modules)
apps/assistant    LangGraph assistant that reads interview records and proposes changes
```

`apps/api` is the product. The other two talk to it over HTTP and WebSocket and can be replaced
without touching it.

### The interview loop

```
candidate mic ──► VAD / turn detection ──► STT (streaming)
                          │
                          ▼
                   state machine ──► LLM (sentence stream)
                          │                    │
                          │                    ▼
                          │              TTS (speech stream)
                          │                    │
                          ▼                    ▼
                    FrameMixer ◄──────── renderer (audio → frames)
                          │
                          ▼
              transport ──► WebSocket and/or WebRTC ──► browser
```

Everything downstream of the state machine carries an **epoch**. That integer is the whole of
cancellation, and §4 explains why it is an integer rather than a kill.

---

## 2. The boundary that matters most

`avatar.contracts` defines every seam as a `Protocol` and **imports nothing from the package**.
It is the only module every layer may depend on, so no layer depends on another.

| Protocol | Implementations |
|---|---|
| `TalkingHeadRenderer` | stub, MuseTalk |
| `Transport` | WebSocket, LiveKit WebRTC, and a tee that drives both |
| `SentenceStream` | scripted, OpenAI-compatible, Anthropic |
| `SpeechStream` | tone, Deepgram Aura |
| `Transcriber` | null, Deepgram nova-3 |

Four rules, each with a test:

1. **The orchestration layer has no ML dependency.** `orchestrator.py`, `mixer.py`, `state.py`
   and `contracts.py` may not import torch, CUDA, or any renderer implementation.
2. **The orchestration layer does not import a renderer.** It receives one.
3. **`contracts.py` imports nothing from the package.**
4. **Importing the orchestration layer pulls in no ML package** — checked by inspecting
   `sys.modules` after the import, not by reading the source.

The payoff is concrete: **all 746 tests run with no GPU, no model weights and no network.** The
state machine, turn-taking policy, mixer cadence, barge-in and scoring are all exercised in a few
seconds on a laptop. The renderer is the only thing that needs a GPU, and it is behind a protocol.

### Why a Protocol is not enough on its own

`runtime_checkable` compares **method names**. Not signatures, and not constructors. That gap is
not theoretical: `AVATAR_RENDERER=musetalk` once raised

```
TypeError: MuseTalkRenderer.__init__() got an unexpected keyword argument 'width'
```

with a fully green suite behind it — and it surfaced as a rejected WebSocket at the instant a
candidate opened their link, because that is when the renderer is first constructed.

`tests/test_renderer_contract.py` closes it: every renderer is built from the server's *own*
option dict (imported, not copied), from no options at all, and from each option alone so a
failure names the key. It also asserts every constructor parameter is keyword-only with a default
— the property that actually makes two renderers substitutable through a seam that only passes
keywords.

---

## 3. The renderer split

The renderer is divided along the line where the hardware requirement starts.

| File | Contains | Needs a GPU |
|---|---|---|
| `renderers/musetalk.py` | windowing, epoch tagging, barge-in reset, the idle loop | no |
| `renderers/musetalk_torch.py` | every torch, CUDA and OpenCV call | yes |
| `renderers/landmarks.py` | 68-point landmarks, the crop box | yes |
| `renderers/stub.py` | a placeholder face driven by audio RMS | no |

So the streaming logic — the part with the interesting bugs — is unit-testable, and the part that
needs 3.7 GB of weights is a lazily-imported implementation detail. `import avatar` never pulls
torch in.

The stub is not a leftover. It is what CI runs, what every other test uses, and the fallback when
weights or a GPU are missing. Work on the real renderer must not be able to break it, which is
what the contract test is for.

---

## 4. Cancellation is an integer

When a candidate interrupts, the avatar must stop. The obvious design — cancel the render task —
is wrong, because a GPU forward pass already in flight cannot be un-started, and awaiting it
before reacting puts model latency on the interruption path.

Instead, every turn has an epoch. Interrupting increments it. **Artifacts from the abandoned turn
are still produced, and die at the consumer** because their epoch is stale.

```
candidate interrupts ──► epoch += 1 ──► state CANCELLING
                                          │
in-flight render completes ──► frames offered ──► mixer rejects: wrong epoch
```

The reaction is a single integer write. The wasted work is bounded by one render window. This is
why `FrameMixer.offer()` takes the current epoch and returns whether it accepted, and why
`reset()` on the renderer abandons buffers without touching a call already inside the backend.

The same reasoning applies to audio the browser has already scheduled: a server-side flush alone
would leave the candidate listening to an abandoned sentence, so the transport sends an explicit
`flush_audio` control message and the client drops what it has queued.

### History is truncated to what was *heard*

The conversation history records how much audio the candidate actually heard, not how much was
sent. The client reports `audio_played` per chunk from Web Audio's own clock. Without this, an
interrupted turn enters history as though it had been delivered in full, and the model's next
question refers to a sentence nobody heard.

---

## 5. Composition instead of configuration

Knowledge retrieval, pronunciation, guardrails and the competency plan are all applied by wrapping
a stream, not by adding branches to the orchestrator:

```python
with_plan(with_guardrail(with_knowledge(llm)))       # SentenceStream → SentenceStream
with_pronunciation(tts)                              # SpeechStream → SpeechStream
```

The orchestrator does not know any of these exist. Each is independently testable, each can be
absent, and the state machine has no feature flags in it.

---

## 6. The state machine

Seven states, and the transition table is the specification:

```
INITIALIZING → IDLE → LISTENING → THINKING → SPEAKING → IDLE
                 ↑                                │
                 └──────── CANCELLING ◄───────────┘
                                                CLOSED
```

Which frame source is shown in which state is a table the mixer reads, not behaviour scattered
across the pipeline — `IDLE` and `LISTENING` show the idle loop, `SPEAKING` shows rendered frames.
That is what makes "what does the candidate see right now" answerable by reading one mapping.

### Standing by is the persona, not a placeholder

The idle loop is built from the reference clip's own frames. The reference *is* the person sitting
still — the upload guidance asks for exactly that — so it is the correct idle loop and nothing has
to be generated. Before this, a candidate saw a grey rectangle, then a face while the interviewer
spoke, then the rectangle again.

The handover needs a frame whose mouth is closed, or it lands mid-vowel and jumps. That comes from
landmarks already computed during enrollment: inner-lip separation, points 62 and 66, normalised by
face height. The quietest quarter of frames is taken rather than anything under a fixed threshold,
because a reference of someone talking never reaches a genuinely closed mouth and an absolute
cut-off would reject a legal reference.

---

## 7. Timing, in one place

Three things must agree on the frame rate: the mixer's cadence, the interval stamped on each
frame, and the `fps` Whisper chunks audio features with. Written down separately they drift, and
the failure is quiet — the mouth slides against the speech slowly enough to read as bad dubbing.

So `TARGET_FPS` lives in `contracts.py` and everything derives from it. `AVATAR_FPS` overrides it.

**Why it is configurable at all** is the interesting part. A renderer that misses its target does
not degrade gracefully — it fails completely, because the mixer drops any frame that misses its
slot. At 3.3 fps against a 25fps clock, every frame is late and every frame is discarded: measured,
three turns produced 169 frames and delivered none, and the candidate watched the placeholder while
the interviewer talked. Running at a rate the hardware sustains is the difference between choppy
video and no video.

A related trap, since it cost real debugging time: the render window must be denominated in
**milliseconds**, not frames. `WINDOW_FRAMES = 16` is 640 ms at 25fps and 2000 ms at 8fps, so
lowering the frame rate silently quadrupled first-frame latency. A latency budget counted in frames
is not a latency budget.

---

## 8. Storage

One backend interface, two implementations, chosen by `AVATAR_STORE`:

| Backend | Shape |
|---|---|
| `Store` (default) | one JSON file per record; runs on a clean clone with no database |
| `PostgresStore` | one table per collection, typed FK columns plus a `doc` JSONB |

The Postgres schema is a hybrid on purpose. Typed columns and real foreign keys where relationships
must be enforced; JSONB for the parts of a record that vary by kind. Updates merge server-side
(`doc = doc || patch`) so a partial write cannot lose a field it never read.

**Media never goes in the database.** Reference clips, thumbnails and recordings are files on disk,
because a 20 MB clip in a JSONB document would be read and rewritten on every unrelated patch to
the same row.

Prepared identities are cached in-process, keyed by reference path, bounded to two entries — each
is roughly a gigabyte of cycled frames, masks and latents. Enrollment that does not warm the thing
it enrolls for is a status field, not a feature, so the cache is module-level and shared by every
renderer instance in the process.

---

## 9. What is deliberately not here

- **Authentication.** None, anywhere. The candidate link is not a credential and the assistant will
  read any transcript in the store. A stated development posture — see [SECURITY.md](SECURITY.md),
  which also explains why real faces change the calculus.
- **A distributed job queue.** Enrollment now returns 202 and runs on a worker thread, claims its
  row with a timestamp, and startup fails anything a dead process left in `preparing` — see
  `avatar/jobs.py`. What is *not* here is a broker. Redis is already running for egress and a queue
  on it would be the conventional answer; at one API process and one GPU it would buy a second
  failure mode and nothing else. The `status` field is the contract a real queue would preserve.
- **Voice cloning at the same time as the face.** A persona *can* sound like a specific person —
  `voices.py`, `tts_clone.py` and the Chatterbox sidecar — but not on one T4 alongside the
  renderer: `avatar_first_frame` goes 3 s to 28 s when both compete. Cloning and a self-hosted
  face are today an either/or, and the sidecar being a separate process is what makes a second
  GPU configuration rather than work.
- **Horizontal scale.** One process, one GPU. The identity cache, the shared model cache and the
  warm-worker argument are all written with a pool in mind, but there is no pool.
