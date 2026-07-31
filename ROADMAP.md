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

### ~~1. Cold start~~ — done
Was **70–150 s** on the first session after a restart. `avatar/warmup.py` loads the models and
prepares the attached faces from the app's lifespan, before traffic is accepted: **2.9 s** now,
verified on the T4. Cost moved to 196.8 s of start-up, where nobody is waiting.

Remaining: a face switch still evicts and re-prepares, because the identity cache holds two.

### 2. A second GPU, or one model at a time
Voice cloning works and is wired in, but the renderer and the cloner cannot share one T4:
`avatar_first_frame` degrades from 2.3–3.0 s to 28–42 s when the sidecar is active. Not memory —
compute contention, because sentence *n+1* is synthesised while frames for sentence *n* render.

The sidecar is already a separate process, so pointing `AVATAR_VOICE_SERVICE` at another host is
configuration. Before spending anything, try a smaller batch size and a lower `AVATAR_FPS` to see
whether the headroom exists on one card.

### 3. The video lags the audio by ~3 s
33–81 frames are discarded per turn, and the discard is correct: `offer()` never drops a late frame,
only a stale epoch, so these are frames still queued when the turn's audio ended. Showing them would
animate a mouth in silence.

The defect is the lag. The renderer starts ~3 s late and has ~8% headroom (8.7 fps against an 8 fps
target), so it cannot catch up inside a turn. Candidates: more headroom via a lower `AVATAR_FPS`, or
less first-frame latency via a shorter render window. **Neither is measured** — two attempts failed
on harness plumbing, and the confounded figures I do have (81 at 8 fps, 46 at 6 fps, both with the
voice sidecar competing) are not worth acting on. Measure this properly first.

### 3. The 2.9× to real time
114.7 ms/frame against 40 ms. The split says where to look, and it is not the U-Net:

| stage | ms/frame |
|---|---|
| VAE decode | 58.8 |
| blend + JPEG (CPU, 4 vCPU) | 43.5 |
| U-Net | 12.4 |

So: a faster decode, and get blending off the critical path — it is CPU work on four cores that
could run concurrently with the next batch. A bigger GPU alone would only shrink the smallest term.

### ~~4. Generative enrollment~~ — done
A photograph uploaded with `animate=true` is animated by LivePortrait before enrollment: 500 frames,
20.0 s, standing-by motion 0.40 mean / 0.69 peak against 0.00 for the same still. Enrollment is a
background job now — `POST /prepare` answers 202 in 0.018 s, claims the row with a timestamp, and
stale rows are reaped at startup.

### 4b. The original ask, for reference
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
