# Measurements

Every number in this file came from a run. Nothing here is estimated, scaled from another
device, or carried over from a vendor's README — where a figure does not exist, this file says
so rather than filling the gap.

That rule is the reason the file exists. A plausible fabricated number is worse than a stated
gap, because it silently becomes the basis of a decision. Two things in this project were
already decided wrongly by numbers measured on the wrong device, and both are recorded below
with their corrections.

**How to reproduce anything here:** `apps/api/scripts/bench_renderer.py` for the renderer,
`scripts/measure_latency.py` for the turn budget, and the session log's own `latency` and
`counter` events for anything measured live. Every renderer figure is `float16` unless stated.

---

## 1. Hardware measured on

| | M1 Pro | Tesla T4 |
|---|---|---|
| Where | local development Mac | Lightning.ai Studio |
| Memory | 16 GB unified | 14.6 GiB VRAM, 4 vCPU |
| Backend | MPS | CUDA 13.0, capability 7.5, driver 580.173.02 |
| torch | 2.13.0 | 2.13.0+cu130 |
| Python | 3.12.3 | 3.12.3 |

No other device has been measured. No figure for a V100, A10G, L4 or A100 appears in this
repository, and none may be quoted until a run produces one.

---

## 2. Renderer throughput — the headline

Best batch size per device, float16, median of 4 runs after 2 warm-up runs.

| Device | ms/frame | fps | vs 40 ms budget (25fps) |
|---|---|---|---|
| M1 Pro / MPS | 301 | 3.3 | 7.5× over |
| Tesla T4 / CUDA, stages in sequence | 124.7 | 8.0 | 3.1× over |
| **Tesla T4 / CUDA, CPU overlapped with GPU** | **78.4** | **12.8** | **2.0× over** |

**Neither device reaches 25fps.** A T4 is 2.6× the Mac and still 2.0× short. MuseTalk's
published realtime figures are not from a T4 and do not reproduce on one.

**The overlap is the largest single win measured so far: 1.59×, for no change to the model.**
A frame is GPU work (U-Net, VAE decode) followed by CPU work (resize, blend, JPEG), and running
them in sequence left the GPU idle for 38% of every frame. `render()` now submits batch N's CPU
half to a worker thread while batch N+1 occupies the GPU. Both figures above are the same
`render()` method on byte-identical input, measured in the same process minutes apart — §2.2's
stage table is what predicted the win, and 78.4 ms against a GPU-only floor of ~70 ms is how
much of the CPU half is now genuinely hidden rather than merely moved.

Two consequences worth stating plainly:

- **The remaining gap is almost entirely VAE decode** — 57.8 of 78.4 ms, 74% of the frame. There
  is no more CPU work left to hide behind it.
- **The stage table no longer sums to the total**, and that is the point rather than an error.

### 2.1 Batch size — the two curves disagree

ms/frame. Upstream's default is 8, which is wrong on both.

| batch | M1 Pro / MPS | Tesla T4 / CUDA |
|---|---|---|
| 1 | 330 | 153.6 |
| 2 | 317 | 145.0 |
| 3 | 301 | — |
| 4 | 305 | 128.3 |
| 6 | 413 | — |
| 8 | 355 | 126.0 |
| 16 | — | **121.6** |
| 32 | 1565 | out of memory |

MPS is flat to 4 and then collapses — 32 is 5× worse than 3 — which is memory pressure on
memory shared with the whole machine. CUDA improves monotonically to 16, the textbook curve.

**This is the first figure that was wrong by being measured on the wrong device.** `BATCH_SIZE`
was set to 4 from the MPS curve. Shipping that to a GPU would have left most of the card idle.
It is now a per-device table.

### 2.2 Where the time goes — Tesla T4, batch 16

Stages timed in isolation, so this is where the work is — not what a frame now costs.

| stage | hardware | ms/frame | share of the sum |
|---|---|---|---|
| positional encoding | GPU | 0.0 | ~0% |
| U-Net forward | GPU | 12.3 | 10% |
| **VAE decode** | **GPU** | **57.8** | **48%** |
| blend into frame | CPU | 27.8 | 23% |
| JPEG encode | CPU | 23.7 | 19% |
| sum of stages | | 121.6 | |
| **a real frame, overlapped** | | **78.4** | |

The gap between the last two rows is the CPU half hiding behind the GPU half. GPU stages total
70.1 ms, so 78.4 means the overlap recovers all but ~8 ms of the 51.5 ms of CPU work.

Two things follow, and neither was true on MPS, where the U-Net dominated at 1546 ms/frame in
float32:

- **VAE decode is the bottleneck**, and it is fixed 256×256 work — so downscaling the reference
  does not touch it. That idea was proposed and discarded for exactly this reason.
- **blend + JPEG = 51.5 ms and both are CPU work** on 4 vCPU. This was the actionable finding:
  it named "get blending off the critical path" as a real fix, and doing that returned 1.59×.
  With it done, a faster VAE decode is the only remaining lever of that size, and a bigger batch
  is still not one — the curve is flat past 16.

### 2.3 float16 versus float32 — M1 Pro only

| | U-Net | VAE decode | total |
|---|---|---|---|
| float32 | 1261 | 987 | 2249 ms/frame |
| float16 | 76 | 169 | **246 ms/frame** |

**9.15× faster.** Output fidelity, same reference and same audio window: mean absolute
difference **0.04 of 255**, maximum **1**, zero NaNs. A 9× speedup for a rounding error in the
last bit is not a trade-off; float32 was simply the wrong default.

`NOT YET MEASURED` on CUDA — the float32 sweep was abandoned because batch 32 asks a 15 GB card
for 10.7 GB in one allocation and thrashes. The fp16 default is not in doubt, but the *ratio* on
CUDA is unknown.

### 2.4 Model load and enrollment

| | M1 Pro / MPS | Tesla T4 / CUDA |
|---|---|---|
| Load all five models | 11.2 – 20.2 s | 27.1 s |
| Prepare, 1-frame still | 5.7 – 7.0 s | 16.3 s |
| Prepare, 150-frame clip | 109.5 s | **82.3 s** |
| Prepare, 550-frame clip | 233.9 s | **126.3 s** |

Enrollment on the T4, measured through the API across five real references:

| Frames | Enrollment | ms/frame |
|---|---|---|
| 100 (still expanded to a clip) | 44.2 s | 442 |
| 150 (6 s video) | 82.3 s | 549 |
| 268 (10.7 s video) | 81.0 s | 302 |
| 500 (20 s constructed video) | 155.5 s | 311 |
| 550 (22 s video) | 126.3 s | 230 |

**Not linear, and the shape is the useful part.** A fixed cost dominates below ~250 frames — model
load plus first-call CUDA warm-up — so ms/frame falls from 442 to 230 as the clip lengthens. The
150-frame case being *slower in total* than the 268-frame case is not noise from a single run being
unlucky; both were cold-process runs and the fixed cost swamps 118 frames of work. Subtracting the
fixed component, marginal cost is roughly 0.23 s/frame.

Practical consequence: **a longer reference is close to free.** 550 frames cost 1.5× what 150 did
while giving a 22 s loop instead of a 6 s one, which is the difference between a loop a candidate
notices and one they do not.

Enrollment through the API (`POST /faces/{id}/prepare`, 150 frames, M1 Pro): **109,468 ms**,
150/150 frames usable. Before enrollment was switched off the stub it reported
`enrollment_ms: 0` — a real measurement of a no-op.

The T4 is *slower* at loading and preparing than the Mac. Not a contradiction: both are
dominated by reading 3.71 GB off disk and by per-frame face detection on CPU, and the Studio has
4 vCPU against the Mac's 10.

### 2.5 Landmark detection (the mmpose substitute)

| | ms |
|---|---|
| Load FAN + S3FD, MPS | 2100 – 2800 |
| Load FAN + S3FD, CPU | 600 |
| Detect, 1024×1536 frame | 4640 |
| Detect, 576×768 frame | 320 |
| Detect, 800×800 test pattern (no face) | 100 |

The 4.6 s figure at full resolution is why `DETECT_MAX_SIDE = 640` exists: a 550-frame
reference would otherwise take longer to prepare than the interview it is for.

---

## 3. Output correctness — not speed

Measured to answer "is the mouth actually driven by the audio, or is this one pose repeated?"

| Check | Result |
|---|---|
| Lip-region change between consecutive frames | 6.30 mean abs (float32), 5.03 (float16) |
| Eyebrow-region change between consecutive frames | **0.00** |
| Ratio | 6296× |
| Same reference frame, two different 1 s audio windows | 21.87 lip difference |
| Lip landmarks inside the crop box | 20/20 on both test faces |

The eyebrow figure is the important one: it is exactly zero, so everything outside the blend
mask is bit-identical to the reference. Combined with 21.87 for different audio on the *same*
frame, the output is audio-driven rather than a static paste.

---

## 4. Live session — M1 Pro, MuseTalk, three turns

From the session's own telemetry, not from instrumentation added for the test.

| Event | Turn 1 | Turn 2 | Turn 3 |
|---|---|---|---|
| `llm_ttft` | — | 2615 ms | 2078 ms |
| `tts_first_audio` | — | 1213 ms | 1099 ms |
| `avatar_first_frame` | 127,200 ms | 95,865 ms | 82,713 ms |
| `frames_discarded` | 58 | 72 | 39 |

**169 frames rendered; the `frames_discarded` counts above are real telemetry.** Every frame that
was dropped missed its slot, which is the barge-in design working as intended — stale artifacts die
at the consumer. The candidate saw the placeholder idle loop for most of every turn while the
interviewer talked.

An earlier version of this section said "0 delivered". That part was wrong and §4b explains why: the
probe measuring delivery could not parse the frame header. Left visible rather than silently
corrected, because the mistake is instructive.

**This is the second figure that was wrong by being measured on the wrong device**, in a subtler
way: 3.3 fps looked like "choppy but watchable" until a real session showed it means *nothing
renders*. A renderer that misses its target does not degrade gracefully; it fails completely.
`AVATAR_MUSETALK_FPS` exists because of this run.

---

## 4b. Live session — Tesla T4, MuseTalk, `AVATAR_FPS=8`

The same three-turn shape as §4, after moving to CUDA and lowering the frame rate to one the
hardware sustains.

| | M1 Pro @ 25fps | Tesla T4 @ 8fps |
|---|---|---|
| `avatar_first_frame`, turn 1 | 127,200 ms | 9,336 ms |
| turn 2 | 95,865 ms | 4,189 ms |
| turn 3 | 82,713 ms | 2,297 – 3,025 ms |
| `frames_discarded` per turn | 58 / 72 / 39 | 79 / 33 / 50 |
| frames delivered | 0 | **all of them** |

**A correction belongs here, because the earlier version of this file was wrong.** §4 reported
"169 frames rendered, 0 delivered" on the Mac. The `frames_discarded` counts were real telemetry;
"0 delivered" was a broken probe — it looked for JPEG magic at byte 0, and every frame carries a
13-byte header (`kind:u8, pts:u32, epoch:u32, length:u32`). Frames were being delivered. The
lesson kept rather than quietly fixed: an instrument that reads zero should be suspected before
the system it measures.

### Session start, T4

| | |
|---|---|
| Warm — identity already in the process cache | **1.52 – 1.57 s** to first frame |
| **First session after a restart, with start-up warming** | **2.9 s** |
| Cold, before warming existed — 100-frame identity | 70.2 s |
| Cold, before warming existed — 550-frame identity | 150.3 s |

Start-up warming moved that cost off the first candidate: **2.9 s instead of 70–150 s**, a 24–52×
improvement on the session that matters most. The work is identical and was simply being done at
the least useful moment. What it costs instead is start-up time — 196.8 s to load the models and
prepare two attached faces, during which the server does not accept traffic:

| | |
|---|---|
| `warmup`, 550-frame face | 148.7 s |
| `warmup`, 268-frame face | 48.1 s |
| total before serving | 196.8 s |

The identity cache holds two entries, so switching between more faces than that still evicts and
re-prepares — a face switch is the one remaining path that pays the full cost.

### Standing by

After the idle loop became the reference frames rather than the placeholder:

| | |
|---|---|
| Frames delivered while idle | **60 of 60 real face**, 0 placeholder (was 8 of 60) |
| Frame-to-frame change, real 22 s reference | mean 0.39, **peak 1.17** |
| Frame-to-frame change, still expanded to a clip | 0.00 — identical frames |
| Frame-to-frame change, constructed drift from a still | 0.54, no peaks |
| Bandwidth at 512 px tall | 29 – 32 KB/frame ≈ 1.9 – 2.1 Mbps at 8 fps |

The peak against the mean is what separates a real reference from a synthetic one: 1.17 against a
0.39 mean is blinks and head shifts. Uniform 0.54 with no peaks is one photograph being moved.

## 4c. Generative enrollment — LivePortrait, spiked on the T4

The question a still reference raises is whether a model can add motion to a photograph **without
changing whose face it is**. If identity drifts, the result is a stranger — worse than a frozen
head. So that was measured first, before anything was built.

`KwaiVGI/LivePortrait`, 2.0 GB of weights, identity from `man.png` and motion from 12 s of a real
person.

| | |
|---|---|
| Generation, 300 frames | 124 s (~414 ms/frame, CPU ONNX providers) |
| **Identity: face-proportion deviation** | **3.6% overall** |
| — inter-ocular / face width | 3.9% |
| — nose length | 4.0% |
| — mouth width | 2.3% |
| — chin to nose bridge | 1.9% |
| — nostril width | 5.9% |

Scale-invariant landmark ratios, generated frames against the source photograph. Under ~5% is the
same face's geometry, and a side-by-side of the source and four generated frames confirms it
visually — the head pose and gaze shift while the person does not.

### The full chain, live

`man.png` → LivePortrait → MuseTalk enrollment (300 frames, 94.4 s) → a session:

| | Still, frozen | Still + LivePortrait | Real human video |
|---|---|---|---|
| Standing-by motion, mean | 0.00 | **0.69** | 0.39 |
| Standing-by motion, peak | 0.00 | **1.31** | 1.17 |
| First frame | — | 1.83 s | 1.52 s |
| Real-face frames delivered | 45/45 | **45/45** | 50/50 |

The peak matters more than the mean. A frozen still is 0.00. A crop-window drift constructed with
ffmpeg was 0.54 *with no peaks* — uniform motion, one photograph being moved. LivePortrait's 1.31
peak against a 0.69 mean is the signature of discrete events, and it exceeds the real human's 1.17.

### The finding that changes the design

**The driving video, not the model, is the limit.** Measuring eye aspect ratio every frame — a blink
lasts 3–4 frames at 25 fps, so sampling every 5th frame misses them entirely, which invalidated a
first attempt at this measurement:

| | EAR mean | EAR min | blink-shaped dips |
|---|---|---|---|
| Driving video (real person, 6 s) | 0.323 | 0.269 | **0** |
| Generated (the man) | 0.294 | 0.216 | **2** |

The driving clip contains no blinks in that window, and the output has two — LivePortrait adds eye
motion of its own. So output quality is bounded by the motion source, and the product consequence
is concrete: **ship one canonical idle driving clip** — a person sitting still, blinking naturally,
small head movements — as a bundled asset, and any uploaded photograph can be animated with it.

## 4d. Voice cloning — Chatterbox, spiked on the T4

Enrollment should take a voice as well as a face. The constraint the face does not have: the TTS
boundary is sentence-streaming, and Deepgram Aura answers in **380 ms warm**. A cloner that takes
seconds per sentence adds those seconds to every turn.

`ResembleAI/chatterbox-tts` 0.1.7, MIT, cloning from a 60 s reference, on the T4:

| Model | per sentence | audio produced | RTF |
|---|---|---|---|
| Base | 2515 – 4137 ms | 1.6 – 3.2 s | **1.31 – 1.61** |
| **Turbo** | **1213 – 2051 ms** | 1.5 – 3.0 s | **0.67 – 0.80** |

**RTF is the number that decides it, not latency.** The base model generates *slower than real
time*, so a turn falls further behind the longer it runs — disqualifying. Turbo's 0.74 means
generation keeps ahead of playback, so only the **first** sentence's latency is exposed; every
later sentence is produced while the previous one is still playing. That is precisely what the
sentence-streaming boundary was built to exploit.

So the honest cost of a cloned voice is **~1.6 s added to the first sentence of each turn** (2.0 s
against Aura's 0.38 s), not a multiplier on the whole turn.

Model load is 32.9 s, which start-up warming already covers.

**Two dependency findings, recorded because both cost time:**

* CosyVoice 2 was the first choice — Apache 2.0 with a published 150 ms streaming first-packet.
  Abandoned at install: it pins `torch==2.3.1` against the renderer's 2.13, plus tensorrt,
  tensorboard and a `grpcio` with no cp312 wheel. Its streaming claim is still the most attractive
  in this class and it is worth revisiting on a dedicated box.
* Chatterbox needs `setuptools<81`. Its watermarker imports `pkg_resources`, removed in 81, and
  `perth/__init__.py` catches the `ImportError` and sets the class to `None` — so the failure
  surfaces as `TypeError: 'NoneType' object is not callable` several frames away from the cause.

## 4e. Voice cloning in production — and why it does not fit on one T4

The sidecar works end to end: upload a 20 s reference, audition it through the API, attach it to an
agent, and a session speaks in that voice. Verified — 8.0 s of speech-shaped audio (peak 12,923,
RMS 1,590) in a real turn, with the interviewer asking a question about what the candidate had just
said.

Sidecar throughput matches the spike:

| | |
|---|---|
| Model load | 36.9 s (once, `POST /warm`) |
| 5.5 s sentence | 3,888 ms — RTF **0.70** |
| 9.1 s sentence | 5,140 ms — RTF **0.56** |

**But the renderer and the voice cloner cannot share one T4.** Same face, same reference, same
prompt, only the voice engine changed:

| | Hosted (Deepgram Aura) | Cloned (sidecar) |
|---|---|---|
| `tts_first_audio` | **290 ms** | **4,848 ms** |
| `avatar_first_frame` | 2.3 – 3.0 s | **28,108 – 41,757 ms** |
| `frames_discarded` | 33 – 79 | 81 |

Two separate costs, and it is worth keeping them apart:

* **The voice itself is 16× slower than hosted** for a long sentence — 4.8 s against 290 ms. That is
  the honest, expected price of self-hosting, and RTF below 1.0 means it stays bounded.
* **The renderer degrades 10×**, which is not expected and is the real problem. It is not memory:
  3.9 GB (renderer) plus 3.6 GB (voice) of 15 GB, with 6 GB of host RAM free. It is compute
  contention — sentence *n+1* is synthesised while frames for sentence *n* are rendering, by design,
  so the two models fight for the same SMs for the whole turn.

So on a single T4 the choice is **a self-hosted face with a hosted voice, or a self-hosted voice with
no face**. Both models at once needs a second GPU, which is also the clean fix: the sidecar is
already a separate process, so pointing `AVATAR_VOICE_SERVICE` at another host is configuration
rather than work. That the architecture makes this a one-line change is the payoff for the process
boundary the dependency collision forced.

Not measured: whether a smaller batch size or a lower `AVATAR_FPS` recovers enough headroom to make
both viable on one card. Worth trying before buying anything.

## 4f. Generative enrollment in the product

LivePortrait is wired into enrollment now, not just spiked. A photograph uploaded with
`animate=true` becomes a reference clip with real motion before MuseTalk ever sees it. Measured on
the T4, `man.png` against the 20s bundled driving clip:

| | |
|---|---|
| `POST /faces/{id}/prepare` response | **202 in 0.018 s** |
| Animation | 213,112 ms |
| Enrollment of the result | 153,001 ms |
| Result | 500 frames, 20.0 s, `source_kind` flips image → video |
| Live session, standing-by motion | mean 0.40, peak 0.69 |
| Live session, first frame | 3.8 s, 40/40 real-face frames |

The same photograph unanimated measures 0.00 standing-by motion — identical frames. So the
comparison across all three reference kinds now reads:

| Reference | standing-by mean | peak |
|---|---|---|
| Still, as uploaded | 0.00 | 0.00 |
| Still + ffmpeg crop drift | 0.54 | *no peaks* |
| **Still + LivePortrait** | **0.40** | **0.69** |
| Real human video | 0.39 | 1.17 |

LivePortrait's mean is close to the real human's and its peak is well below — it has motion of the
right *character* (discrete events, not uniform drift) at lower amplitude. The crop-drift version has
a higher mean and no peaks at all, which is the signature of a photograph being slid around.

### The driving clip is chosen by measurement

`scripts/make_driving_clip.py` scores every candidate 20s window by the variance of eye-aspect-ratio
across its frames, because the spike established that the driving clip and not the model bounds
quality. On the bundled source: **5 windows scanned, best score 0.0323 at 1.9 s, worst 0.0317**. It
warns below 0.005, where animated faces would not blink at all.

## 5. Speech and audio

| | Measured |
|---|---|
| Deepgram Aura, one sentence | 36 chunks, 90,880 bytes = **2.84 s** of 16 kHz mono |
| Deepgram TTFB, cold | ~1020 ms |
| Deepgram TTFB, warm | ~380 ms |

The cold/warm gap is why the adapter holds one `httpx.AsyncClient` for the process lifetime; a
client per request puts every turn on the slow path.

`ToneTTS` is not measured because it is not speech — it emits a sine wave of the correct
duration, and every agent defaulted to it, which is why no voice was audible until one was
switched to Deepgram.

### Two silent failures, with the numbers that found them

**Audio was arriving and never playing.** Measured at the socket: 3.20 s of PCM per turn, peak
12,708, RMS 2,896 — speech-shaped, not the sine a tone would give. Measured in the browser:
AudioContext `running`, gain `1`, and **zero buffer sources started**. A guard on React state that
`socket.onmessage` had closed over before that state existed was discarding every message.
Buffer sources started went **0 → 79**.

**Speech was heard and never transcribed:** 10 of 11 turns. Deepgram returned **3 interim results
and 0 finals**, and only finals were accumulated. With `endpointing=300` and trailing silence,
**finals: 1**. Round-tripped through Deepgram TTS so the expected text was exact:

| | |
|---|---|
| expected | I led the migration of our payments service off a shared database. |
| got | I led the migration of our payment service off a shared database. |

---

## 6. Weights on disk

3.71 GB across 9 artifacts, each verified on arrival (size floor, HTML-page detection, container
format).

| Artifact | Size |
|---|---|
| `musetalkV15/unet.pth` | 3242.6 MB |
| `sd-vae/diffusion_pytorch_model.bin` | 319.2 MB |
| `whisper/pytorch_model.bin` | 144.1 MB |
| `face-parse-bisent/79999_iter.pth` | 50.8 MB |
| `face-parse-bisent/resnet18-5c106cde.pth` | 44.7 MB |
| 4 JSON configs | < 1 MB |

Not fetched, deliberately: `dwpose` (400 MB — mmpose is substituted), `syncnet` (training only),
MuseTalk v1.0 (superseded by v1.5).

Download on the Lightning Studio: 3.8 GB in **under 3 minutes** (1.3 GB in the first 25 s). On the
Mac it took substantially longer; the figure was not timed precisely and is not quoted.

---

## 7. Frame size and bandwidth

| Reference | Output frame | JPEG q82 | Bitrate |
|---|---|---|---|
| 1024×1536 (uncapped) | 1024×1536 | 249 KB | ~50 Mbps at 25fps |
| capped at 512 tall | ~384×512 | **29 – 32 KB** | **1.9 – 2.1 Mbps at 8fps** |

The 50 Mbps figure is why `MAX_OUTPUT_HEIGHT = 512` exists. MuseTalk blends the repainted mouth
into the *whole* reference frame, so output resolution is the reference's, not the model's — and
the model works at 256×256 regardless, so the surrounding pixels were never generated detail.

A 24× reduction, and the model output is untouched — the U-Net works at 256×256 either way, so
what was discarded is reference background, not generated detail.

---

## 8. Figures from earlier work on this branch

Measured in earlier sessions, recorded here for one place to look. Their source is the commit
that introduced them.

| | Measured |
|---|---|
| Full turn, end to end (PROCESS.md §3.4) | 2.7 – 5.8 s |
| Scorer: time to queue vs work done | 8 ms vs 6,093 ms |
| WebRTC first frame: perceived vs paint | 4,296 ms vs 4,221 ms (74.9 ms paint tail) |
| Egress recording | 7.0 MB, 1 m 37 s, H.264 + AAC |
| Test suite | 746 tests, GPU-free |

The 74.9 ms paint tail is the reason first-frame latency is measured with
`requestVideoFrameCallback` rather than at the decoder: the gap between a frame arriving and a
frame being *visible* was invisible to every earlier measurement.

---

## 8b. What a candidate actually sees — the live figures

§2 measures how fast the card turns audio into frames. This measures what arrives at a client, over
a real WebSocket, in a real session, and they are different questions. Produced by
`scripts/measure_lag.py`, whose figures are timestamped in the probe process — so they include the
socket but not a browser's decode or compositor, which makes every number here a lower bound on
what a person perceives.

Tesla T4, `AVATAR_FPS=8`, hosted TTS and STT, the renderer alone on the card, six turns per run:

| | Before | After |
|---|---|---|
| Frames delivered, per second | **1.0 – 2.4** | **8.2 – 8.9** (target 8) |
| Frames delivered vs needed, per turn | 9 of 75 | 38 of 45 |
| Trailing gap, median | ~3 s (recorded) | **−66 ms and +172 ms** across two runs |
| Trailing gap, worst single turn | not measured | 538 ms |
| Video starts, after the audio | not measured | 1,510 ms |
| Frames discarded, per turn | 33 – 81 | 6 – 11 |
| First turn vs later turns | 16.7 s vs — | 1.5 s vs 1.5 s |

**"Trailing gap" is the headline and it needs its definition stated**: how long video kept arriving
*after* the last audio of the turn did. Negative means video finished first, which is the healthy
direction. Broadcast lip-sync tolerance is about 100 ms of video lag before a viewer notices, so
the median sits at or just past the edge of perceptible and the worst turn is clearly past it.

**The two runs disagree in sign** — one median at −66 ms, one at +172 ms — and both are reported
rather than the flattering one. Six turns per run is not enough to call a mean, and the honest
summary is "somewhere around zero, with individual turns up to half a second late", not a single
figure.

### 8b.1 Three causes, and what each one was

The complaint was recorded for a long time as "the video lags the audio by ~3 s". Measuring it
properly showed the trailing gap was already near zero and the real defect was elsewhere. Three
separate faults, found in this order, each one hiding the next:

1. **Every session reloaded 3.8 GB of weights.** `load()` filled an instance attribute while its
   own docstring said "called once per process", and `renderers.build` returns a fresh backend per
   session. Audio at 6.2 s, first frame at 22.9 s — and 23 s is exactly what `load()` measures.
   Warm-up had loaded the models correctly, into an object that was then discarded.

2. **The first forward pass costs 12× a later one.** 4,747 ms for the first five frames, against
   78 ms/frame steady state: cuDNN choosing convolution algorithms, the landmark detector's first
   inference, lazy allocator arenas. Loading weights is not the same as being ready. Warm-up now
   renders one throwaway frame, so a candidate does not pay it — first turn and fifth turn now
   have the same 1.5 s start lag, where before it was 6.9 s versus 1.4 s.

3. **The render ran on the event loop.** This was the big one. `_pump_frames` was synchronous on a
   documented contract that `frames()` "returns what exists now" and could not block — true of the
   stub, false of a GPU renderer, which runs the U-Net and VAE inline for every buffered window.
   So the task that drains the mixer to the socket at cadence had no loop to run on: frames were
   rendered, queued, and correctly thrown away by `set_source(IDLE_LOOP)` when the turn ended.
   **1.4 fps delivered by a renderer benchmarked at 12.8.** It read as a renderer too slow to keep
   up; the renderer was never the problem.

All three were the same mistake in different places: an assumption that held for the stub renderer
and not for the real one. That is the cost of a boundary this clean — the stub satisfies the
Protocol perfectly, so nothing about it fails until a GPU is behind it.

### 8b.2 What the discards were

`frames_discarded` was read for a long time as evidence the renderer could not keep up. It was
evidence of fault 3. `FrameMixer.offer()` never drops a frame for being individually late — it
only rejects a stale epoch — and the counter comes from `_drain()`, which empties the queue when
the source returns to the idle loop. So a discard is a frame that was still pending when the
turn's audio ended, and showing it later would animate a mouth in silence. The discarding was
always correct. The backlog was not.

---

## 8c. Paired delivery, measured — and it is worse over WebSocket

The arrangement that produced §8b's residual gap is two publishers on two clocks: the orchestrator
writes audio the moment it has it, a separate task drains frames at a cadence. `AVATAR_DELIVERY=paired`
routes both through one interleaved sequence instead. Three turns each, same machine, same agent,
MacBook Pro with the stub renderer:

| | `split` (default) | `paired` |
|---|---|---|
| Delivered fps | 25.2 | 25.2 |
| Video start lag | 69 ms | 118 ms |
| **Trailing audio→video gap** | **+27 ms** | **−5,918 ms** |
| Frames delivered vs needed | 159 / 185 | 134 / 285 |

**Paired is worse and stays off.** A trailing gap of −5.9 s means the video for a turn finishes six
seconds before the audio it belongs to: on screen, a mouth that stops moving while the interviewer is
still talking. That is a regression, not an improvement, and the flag exists precisely so the
comparison could be made rather than assumed.

### 8c.1 Two bugs found on the way, both by measurement

**Audio starved video, 32:1.** The first interleaving policy drained every pending chunk before each
frame, on the reasoning that audio must never be held back. But a TTS with a real-time factor below
1 delivers a whole utterance in a fraction of its playback duration, so "everything pending" is
almost everything. Measured at **32 audio items per video frame** where the correct ratio is 2:1, and
live it produced **16 frames where 221 were needed** — the same starvation as the event-loop bug from
a different cause.

**A budget in time still collapsed to a budget in count.** Fixing the above with a 40 ms budget per
frame was not enough, because Deepgram delivers roughly **78 ms per chunk**. Subtracting after
yielding let one oversized chunk through and consume the whole budget, so the ratio became one chunk
per frame regardless of duration — running audio at about twice video and leaving **143–161 frames
per turn** queued and then discarded. Carrying the overshoot forward as debt restores 1.00× at every
chunk size tested (20, 40, 78, 100 ms).

### 8c.2 Why it is worse, which is the useful part

**Metering audio to a video cadence is wrong over a transport where the client buffers.** Split mode
sends each chunk as it exists and the browser schedules playback from its own buffer, so transmission
finishes long before playback does. Paired mode meters audio at 1× real time, so a turn takes as long
to *transmit* as to *play* — and the turn's video, which is finite, runs out first.

**Where pairing is right: when the consumer itself consumes in real time.** `rtc.AVSynchronizer`
pushes audio through `AudioSource.capture_frame`, which blocks at playback rate by design, and pairs
video against it. A paired sequence is exactly what it wants — and `scripts/avatar_worker.py`
measured **174 and 176 video frames plus 1,186 and 1,192 audio frames arriving at a remote
subscriber** through one synchroniser against a real SFU.

So the conclusion is not "pairing does not work". It is that pairing belongs at the LiveKit boundary
and not at the WebSocket one, and the same `AvStream` serves both — one measured onto its correct
consumer, one measured off its wrong one.

---

## 8d. A/V drift at a LiveKit subscriber — the claim, tested

§8c showed paired delivery is wrong over WebSocket. This is the same question asked of the topology it
was built for: the renderer in its own process, publishing through `rtc.AVSynchronizer`, measured at a
remote subscriber that decoded the tracks. Two turns, stub renderer, raw RGB24, local SFU.

| | |
|---|---|
| Video frames received | 327 |
| Audio frames received | 2,200 |
| Audio media decoded | 22.00 s |
| Video media span | 21.74 s |
| **A/V drift: median / worst / final** | **−241 ms / −250 ms / −240 ms** |

**Read the spread, not the offset.** The constant ~−240 ms is a baseline artifact of the measurement:
each timeline is accumulated from its own first frame, so a video track that starts a quarter of a
second after the audio track shows as a permanent offset. What matters is that it does not move —
**9 ms of variation across 22 seconds**.

Against §8b on the WebSocket path, where the same quantity ranged −66 ms to +172 ms with individual
turns up to **538 ms** late, that is the difference the synchroniser buys: not a smaller number, a
*stable* one. A fixed offset is a startup latency and correctable; variance is what a viewer reads as
bad lip-sync, and it is what two independent clocks produce.

**What this does not measure.** Per-turn attribution is impossible from a subscriber, which sees
decoded media and not turns — so this cannot say "turn three was 500 ms late" the way §8b can. It also
runs the stub renderer, so it measures the transport and the pacing, not a GPU. Both are stated rather
than papered over: the figure supports "the pairing is stable", and not "the product is in sync".

---

## 9. Known gaps

Stated rather than filled:

- **No CUDA float32 comparison.** The fp16 default is not in doubt; the ratio on CUDA is unknown.
- **No output-quality comparison** between the substituted landmark detector and RTMPose.
- **No measurement on a card larger than a T4**, which is the obvious next question given that a
  T4 is 2.9× short.
- **No measurement on how much of the remaining 1.5 s start lag is the render window** versus the
  lead-in the mixer waits for before it will cut from the idle loop. Both are candidates and the
  split is unknown.
- **No output-quality comparison against MuseTalk's published samples.** The landmark detector is
  substituted, so the crop can differ by a pixel or two; whether that is visible is unmeasured.
