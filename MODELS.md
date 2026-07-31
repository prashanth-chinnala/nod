# Models

Every model this product runs, why it was chosen over the alternatives, and what would replace
it. Written so that a model choice can be revisited without re-deriving the reasoning — and so
that "we picked the popular one" is never the answer to why something is here.

The measured figures behind the arguments are in [MEASUREMENTS.md](MEASUREMENTS.md). Where a
figure does not exist, this file says so rather than borrowing a vendor's.

---

## 1. The constraint that decides everything

**A conversation cannot be pre-rendered.** The reply does not exist until the candidate has
finished speaking, so frames must be produced *after* that and within about a second. This is not
a performance target we chose; it is the shape of the problem.

That single constraint eliminates most of the impressive work in this field. Models that generate
video — including the ones that generate *better* video than anything here — need the whole
utterance up front and produce a file. They are the right tool for a different product.

The second constraint is hardware: **one NVIDIA T4, 15 GB, 4 vCPU** (Lightning Studio). Several
credible 2026 models need multi-GPU H800-class machines. Those are not options today, and saying
so is more useful than listing them as candidates.

---

## 2. Face — MuseTalk v1.5

Five models cooperate to put a talking face on screen.

| Component | Model | Size | Job |
|---|---|---|---|
| Lip-sync core | `musetalkV15/unet.pth`, `UNet2DConditionModel` | 3.24 GB | repaints the mouth in latent space |
| Image codec | `stabilityai/sd-vae-ft-mse`, `AutoencoderKL` | 319 MB | 256×256 crop ↔ 32×32×4 latents |
| Audio encoder | `openai/whisper-tiny`, encoder only | 144 MB | conditions the U-Net |
| Blending mask | BiSeNet + ResNet-18 | 95 MB | feathers the generated mouth in |
| Landmarks | `face_alignment` 1.5.0 (FAN + S3FD) | ~180 MB | **substituted**, see §5 |

### It is not a diffusion model in operation

This matters more than the architecture name suggests. The renderer passes `timestep=0` and there
is no scheduler, no `num_inference_steps`, no sampling loop anywhere in it — **one forward pass
per batch of frames.** That is the entire reason it can approach real time when a real diffusion
model cannot.

Two details in its config say what it really is. `in_channels: 8` against `out_channels: 4` — it
consumes two concatenated latents, the masked lower face and the intact reference, and predicts
one. And `cross_attention_dim: 384` is exactly whisper-tiny's `d_model`, so audio enters where a
text prompt would in an image model.

So the honest description is **audio-conditioned latent inpainting**. It never synthesises a
person. It repaints the mouth region of the reference frames, every frame — which is why the
reference is read at every frame, why the output *is* the operator's clip with a new mouth, and
why a still reference can never blink.

### Is it outdated?

**No, for this use, and the question deserves a real answer rather than a reassurance.**

The 2026 field splits into three classes:

| Class | Examples | Verdict here |
|---|---|---|
| Single-step lip-sync onto existing frames | **MuseTalk**, LatentSync, Wav2Lip | the only class that fits |
| Portrait animation from a still | Hallo2, EchoMimic, SadTalker, MuseV | enrollment only — see §4 |
| Streaming video diffusion | LiveAvatar (14B, 45fps on multi-card H800), StreamAvatar | correct future, wrong hardware |

Within the first class, the trade is well established: **LatentSync** (ByteDance) renders the
mouth interior, teeth and tongue more sharply; **MuseTalk** is faster; **Wav2Lip** has the
tightest sync and the worst resolution. MuseTalk is the speed/quality balance, which is the axis a
live conversation actually cares about.

`LiveAvatar` is the direction this field is going, and it is honest to say it is better. It is also
a 14-billion-parameter model reported at 45fps on *multi-card H800*. On one T4 it is not a
candidate. Designing toward it is reasonable; building on it is not.

### The one claim we cannot reproduce

MuseTalk's paper reports **30+ fps at 256×256**. We measure **8.7 fps** on a T4 (batch 16,
float16). That gap is not a defect in either place, and the per-stage split explains most of it:

- The U-Net — the part the paper is about — is **12.4 ms/frame**, comfortably real time.
- **VAE decode is 58.8 ms** and **CPU blending plus JPEG encode is 43.5 ms**, together 89% of our
  frame budget. A paper measuring the model does not include our blending and encoding, and a T4
  is not the card those figures come from.

We quote our own number with our own hardware attached to it. Nobody should read 30fps into this
product.

---

## 3. Speech and language

| Role | Model | Hosted | Why |
|---|---|---|---|
| Text to speech | Deepgram Aura 2 (`aura-2-thalia-en`) | yes | measured 380 ms warm TTFB; streams sentence-by-sentence |
| Speech to text | Deepgram `nova-3` | yes | streaming with endpointing, which the turn boundary depends on |
| Language | `gpt-oss:20b` via an OpenAI-compatible endpoint | yes | open-weight, self-hostable later without a code change |

The LLM adapter speaks the OpenAI wire format deliberately: Ollama, vLLM and LM Studio all do
too, so moving this in-house is a base URL, not a rewrite.

### Voice cloning — spiked, viable, not yet wired in

Deepgram cannot clone. For a persona that sounds like a specific person the model is
**`ResembleAI/chatterbox-tts`, Turbo variant** — MIT, 350M, self-hosted.

The deciding measurement is not latency but **real-time factor**. The base model runs at RTF
1.31–1.61: slower than real time, so a turn falls further behind the longer it speaks, which is
disqualifying regardless of how good it sounds. Turbo runs at **0.67–0.80**, so generation keeps
ahead of playback and only the first sentence of a turn exposes its latency — roughly 2.0 s against
Aura's 0.38 s. Figures in [MEASUREMENTS.md](MEASUREMENTS.md) §4d.

Rejected: **CosyVoice 2**, despite the best claim in the class (Apache 2.0, 150 ms streaming
first-packet). It pins `torch==2.3.1` against the renderer's 2.13, plus tensorrt, tensorboard and a
`grpcio` with no cp312 wheel. Worth revisiting on a dedicated box, where its streaming design would
beat generating a whole sentence at a time. **F5-TTS** was rejected for the structural reason: it
generates an utterance rather than streaming, so it cannot exploit the sentence boundary at all.

---

## 4. The gap a still image cannot cross

Upload a photograph and the persona holds one pose. Only the mouth moves; the head never does,
and it never blinks. This is not a bug to fix in the renderer — it is what "repaint the mouth of
the frames you were given" means when there is only one frame.

The fix belongs at **enrollment**, not on the live path: generate a reference *clip* with real
motion once, offline, then let MuseTalk lip-sync that clip live. Nothing about serving changes.

Candidates, cheapest first:

| Model | Kind | Fits one T4 |
|---|---|---|
| SadTalker | 3DMM-based portrait animator | comfortably |
| MuseV | MuseTalk's own designed companion | likely, with care |
| Hallo2 / EchoMimic | diffusion portrait animation | tight |
| LongCat-Video-Avatar-1.5 | audio-driven generation, MIT | **no — needs 2 GPUs** |

Because this stage is offline, the hardware constraint is soft: renting a larger GPU for the
minutes one face takes is legitimate, and keeps the T4 for serving.

**Resolved by a spike, and the answer picked the model: `KwaiVGI/LivePortrait`.**

Identity preservation is its design rather than a hoped-for property — implicit keypoints transfer
expression and head motion from a driving video onto a source portrait, and it explicitly does not
swap faces. Measured on the T4: **3.6% face-proportion deviation** from the source photograph, and
a visual side-by-side confirms the same person across generated frames. The full chain renders live
— photo → LivePortrait → MuseTalk → browser, 45/45 real-face frames, standing-by motion peaking at
1.31 against a real human's 1.17 and a frozen still's 0.00. Figures in
[MEASUREMENTS.md](MEASUREMENTS.md) §4c.

It also beat the alternatives on fit, not just results. SadTalker and the diffusion animators
*generate* motion from audio or noise; LivePortrait *transfers* it from a clip we choose, which is
why identity survives and why the output is controllable. 2.0 GB of weights, comfortable on a T4,
414 ms/frame with CPU ONNX providers — irrelevant for work that happens once per face.

**The limit is the driving video, not the model.** Measured every frame, the driving clip contained
zero blinks in the window tested and the output contained two — the model adds eye motion, but
output quality is bounded by the motion source. The product consequence: ship one canonical idle
driving clip as a bundled asset, and any uploaded photograph can be animated with it.

An interim measure exists and its limits should be stated plainly: a reference clip can be
constructed from a still by drifting a crop window with two non-harmonic sine terms. Verified
moving — frame 0 versus frame 120 differs by 14.27, and standing-by frames change by 0.54 mean —
but **no model is involved**. It is one photograph being moved. It reads as a webcam wobble, it
does not blink, and the expression never changes.

---

## 5. The one substitution we made, and why

MuseTalk's `preprocessing.py` builds an mmpose RTMPose model at *import* time. That arrives as
`mmcv==2.0.1` + `mmpose==1.1.0`: pinned, compiled, Linux/Windows wheels only, against a torch
generation several releases old. It is what killed the first spike run, and on Apple Silicon it
does not build at all.

What it is used *for* is one line — `keypoints[0][23:91]`, the 68 face keypoints of the
COCO-WholeBody skeleton, which is the iBUG-68 layout. `face_alignment` has produced exactly that
layout since 2017 from a pure-torch FAN, on CUDA, MPS or CPU, with no compiled ops. The bounding
box upstream pairs it with already came from S3FD *vendored out of face_alignment*.

So the substitution replaces half a file's worth of pinned dependency with the library the other
half was copied from. Upstream's crop arithmetic is transcribed unchanged — every constant, the
`max(0, ...)` floor, the degenerate-box fallback — because the U-Net was trained on crops that
arithmetic produced.

**The honest cost:** the landmarks are not bit-identical to RTMPose's, so the crop can differ by a
pixel or two, so output may differ from MuseTalk's published samples. That is unmeasured. It is
also the only reason this runs on a development machine at all.

---

## 6. Why any of this is replaceable

Model choice here is reversible, and that is a deliberate property rather than a happy accident.

- `avatar/contracts.py` defines `TalkingHeadRenderer` as a protocol and imports nothing from the
  package. Every renderer is selected by `AVATAR_RENDERER` at runtime.
- `renderers/musetalk.py` holds the streaming logic — windowing, epoch tagging, barge-in reset —
  and imports no torch. `renderers/musetalk_torch.py` holds every CUDA call. The split is
  enforced by `tests/test_boundaries.py`, not by convention.
- `tests/test_renderer_contract.py` constructs every renderer from the server's own option dict.
  It exists because a `runtime_checkable` Protocol compares method *names* — not signatures, and
  not constructors — so `AVATAR_RENDERER=musetalk` once raised `TypeError: unexpected keyword
  argument 'width'` with a fully green suite behind it, at the moment a candidate opened their
  link.

The practical consequence: **adding LatentSync is a new file behind the same boundary and an A/B,
not a migration.** That is the argument for having spent effort on the seam rather than on the
model.

---

## 7. What we would change, and when

| Trigger | Change |
|---|---|
| A GPU larger than a T4 becomes normal | re-measure the batch curve; revisit LatentSync for fidelity |
| Multi-card H800-class hardware | LiveAvatar or StreamAvatar class, and the live path is redesigned |
| Identity preservation proves out on a portrait animator | generative enrollment, so a photo yields a persona that blinks |
| A persona must sound like a specific person | self-hosted voice cloning as a second enrollment input |
| Everything must run in-house | LLM base URL moves; STT and TTS need self-hosted replacements, which is real work |
