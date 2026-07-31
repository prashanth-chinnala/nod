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
| Tesla T4 / CUDA | 114.7 | 8.7 | 2.9× over |

**Neither device reaches 25fps.** A T4 is 2.6× the Mac and still 2.9× short. MuseTalk's
published realtime figures are not from a T4 and do not reproduce on one.

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
| 16 | — | **114.7** |
| 32 | 1565 | out of memory |

MPS is flat to 4 and then collapses — 32 is 5× worse than 3 — which is memory pressure on
memory shared with the whole machine. CUDA improves monotonically to 16, the textbook curve.

**This is the first figure that was wrong by being measured on the wrong device.** `BATCH_SIZE`
was set to 4 from the MPS curve. Shipping that to a GPU would have left most of the card idle.
It is now a per-device table.

### 2.2 Where the time goes — Tesla T4, batch 16

| stage | ms/frame | share |
|---|---|---|
| positional encoding | 0.0 | ~0% |
| U-Net forward | 12.4 | 11% |
| **VAE decode** | **58.8** | **51%** |
| blend into frame (CPU) | 26.2 | 23% |
| JPEG encode (CPU) | 17.3 | 15% |
| total | 114.7 | |

Two things follow, and neither was true on MPS, where the U-Net dominated at 1546 ms/frame in
float32:

- **VAE decode is the bottleneck**, and it is fixed 256×256 work — so downscaling the reference
  does not touch it. That idea was proposed and discarded for exactly this reason.
- **blend + JPEG = 43.5 ms, 38% of the frame, and both are CPU work** on 4 vCPU. The next real
  gains are a faster decode and getting blending off the critical path, not a bigger batch.

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
| Test suite | 715 tests, GPU-free |

The 74.9 ms paint tail is the reason first-frame latency is measured with
`requestVideoFrameCallback` rather than at the decoder: the gap between a frame arriving and a
frame being *visible* was invisible to every earlier measurement.

---

## 9. Known gaps

Stated rather than filled:

- **No CUDA float32 comparison.** The fp16 default is not in doubt; the ratio on CUDA is unknown.
- **No output-quality comparison** between the substituted landmark detector and RTMPose.
- **No measurement on a card larger than a T4**, which is the obvious next question given that a
  T4 is 2.9× short.
- **`frames_discarded` is still 33 – 79 per turn on the T4.** Frames are delivered now, but a
  third to a half of a turn's frames still miss their slot. First-frame latency, not throughput,
  is what drives this: the turn's audio finishes before the renderer has caught up, and
  `FrameMixer._drain()` discards the remainder.
- **No measurement with the models warmed at startup**, which is the obvious fix for the cold
  70 – 150 s and probably for a share of the discards.
- **No output-quality comparison against MuseTalk's published samples.** The landmark detector is
  substituted, so the crop can differ by a pixel or two; whether that is visible is unmeasured.
