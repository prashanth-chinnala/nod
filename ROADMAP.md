# Roadmap

What is built, what is next, and what was deliberately left out. Ordered by whether it would be
felt by someone using the product, not by how interesting it is to build.

Every measured claim here is from [MEASUREMENTS.md](MEASUREMENTS.md).

---

## Built and verified

Verified means driven end to end and observed — in a browser, or over a real socket with the output
inspected — not "the code exists and the tests pass".

### The interview
- **Two-way conversation** over WebSocket and, where an SFU exists, WebRTC. Streaming
  transcription, barge-in, per-agent turn detection.
- **Barge-in by epoch.** Cancellation is an integer write; artifacts from an abandoned turn are
  produced and die at the consumer. History truncates to audio the candidate *acknowledged*
  playing, not audio that was sent.
- **A competency plan** that changes what gets asked and tracks coverage, applied by wrapping the
  sentence stream rather than by branching the orchestrator.
- **Knowledge retrieval, pronunciation, guardrails** — same composition pattern, all optional.
- **Async scoring.** 8 ms to queue against 6,093 ms of work, so it is off the interview's critical
  path. Labelled verdicts, quotes verified against the transcript, and `unavailable` rather than a
  fabricated total when the judge fails.

### The face
- **MuseTalk v1.5 on CUDA**, producing a real audio-driven face. Verified: the lip region changes
  6.30 between frames while the eyebrow region changes **0.00**, and the same frame under two
  different audio windows differs by 21.87.
- **Enrollment from a video or a photo**, validated by reading the file with `ffprobe`. Refuses
  what cannot work, warns about what will work badly, and says which.
- **The persona stands by**, using the reference's own frames as the idle loop. Handover lands on a
  closed mouth, chosen from inner-lip separation in the landmarks enrollment already computes.
- **A verified weights fetch** that replaces upstream's silent-failure script.
- **mmpose removed** and replaced with a pure-torch landmark detector, which is what makes the
  renderer installable outside a Linux CUDA box at all.

### The product around it
- **A console** for agents, faces, rubrics, knowledge, tools, guardrails, pronunciations, sessions
  and reports. Resource references are pickers; provider-owned identifiers are suggestion combos,
  because a closed list would make a model released tomorrow unselectable.
- **Postgres**, one table per collection with typed foreign keys plus a `doc` JSONB, and a
  server-side merge so a partial write cannot lose a field it never read. The JSON-file store
  remains the credential-free default.
- **Egress recording** — a real 7.0 MB, 1m37s H.264 + AAC file.
- **First-frame latency measured to paint** with `requestVideoFrameCallback`, which exposed a
  74.9 ms tail that every earlier measurement missed.
- **A screen-aware assistant** with speech in and out, read/write tool separation, and attributed
  proposals a human commits.
- **715 tests, GPU-free**, with the boundary enforced by inspecting `sys.modules` rather than by
  convention.

---

## Next

### 1. Cold start — the worst number a candidate can hit
The first session after a restart waits **70–150 s** while models load and the attached face
enrolls. Warm both at startup: load the models, then prepare every face attached to an agent. The
identity cache already exists and is process-wide; nothing fills it until someone arrives.

This is first because it is the only item on this list that a candidate experiences as the product
being broken.

### 2. Frames still discarded — 33 to 79 per turn
Frames are delivered now, but a third to a half of a turn's frames still miss their slot. The cause
is first-frame latency, not throughput: the turn's audio finishes before the renderer catches up and
`FrameMixer._drain()` discards the remainder. Warming (item 1) should take a bite out of this;
measure before doing anything cleverer.

### 3. The 2.9× to real time
114.7 ms/frame against 40 ms. The split says where to look, and it is not the U-Net:

| stage | ms/frame |
|---|---|
| VAE decode | 58.8 |
| blend + JPEG (CPU, 4 vCPU) | 43.5 |
| U-Net | 12.4 |

So: a faster decode, and get blending off the critical path — it is CPU work on four cores that
could run concurrently with the next batch. A bigger GPU alone would only shrink the smallest term.

### 4. Generative enrollment, so a photo can blink
A still reference holds one pose forever; that is what "repaint the mouth of the frames you were
given" means with one frame. The fix is offline: generate a reference clip with real motion at
enrollment, then lip-sync it live. Candidates and the open question — identity preservation — are in
[MODELS.md](MODELS.md) §4.

**Spike before building.** MuseTalk was written entirely against its README and every signature was
wrong; a batch size derived on MPS was backwards on CUDA. The same discipline applies here.

### 5. A real job queue
Enrollment is synchronous HTTP taking minutes, and nothing reaps a row a crash left in `preparing`.
That is tolerable now and it is the first hard blocker for item 4. Redis is already running for
egress.

### 6. Authentication
See [SECURITY.md](SECURITY.md). There is none, anywhere, by documented choice — and that choice was
made when references were vendor demo assets. A store of real people's faces and voices is
biometric data, and this stops being deferrable the moment the first real face is uploaded.

---

## Deferred, with reasons

| Item | Why not now |
|---|---|
| Streaming video diffusion (LiveAvatar class) | 14B parameters, reported at 45 fps on multi-card H800. The right direction; the wrong hardware |
| LatentSync as the renderer | Sharper mouth interior, slower. Cheap to try behind the existing boundary — an A/B, not a migration |
| Voice cloning | A second enrollment input and a different model. Kept off the face pipeline's critical path deliberately |
| Self-hosted STT and TTS | What makes "no candidate data leaves" true rather than aspirational. Real work, not configuration |
| Embedding-based retrieval | Only when BM25 is measurably insufficient. Adds a dependency and a network hop to a budget with none spare |
| Warm renderer pool, horizontal scale | One process, one GPU today. The identity cache is written with a pool in mind, but there is no pool |
| Pre-rendering scripted moments | Greeting, opening question and sign-off are predictable and could be rendered once per face. Cheaper than item 4 and attacks item 1 — worth doing before either |

---

## Known gaps in what is built

Not future work — things that are wrong or missing in shipped paths.

- **Silence re-prompt turns are never recorded.** A turn record only opens on a `heard` event, and
  `on_idle_tick` calls `_begin_turn()` without one, so a re-prompt after silence leaves no trace.
- **No CUDA float32 comparison**, so the size of the float16 win on CUDA is unknown. The default is
  not in doubt; the ratio is.
- **No output-quality comparison** between the substituted landmark detector and RTMPose. The crop
  can differ by a pixel or two; whether that is visible is unmeasured.
- **The `honesty` audit dimension never ran** — three consecutive API failures — so that pass is
  incomplete.
- **Calibration anchors have no rubric field.** The assistant can propose one and verifies the
  quote, but there is nowhere to promote it into.
- **Deleting a face leaves its media.** The clip, thumbnail and generated still-clip all survive.
