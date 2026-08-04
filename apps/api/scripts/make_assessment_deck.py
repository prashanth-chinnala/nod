#!/usr/bin/env python3
"""
Build the assessment presentation: how a conversational avatar system works, and what we built.

**Who this is for and how it differs from `make_deck.py`.** That one is a run sheet for demoing
software. This one is for presenting to the people who set the brief: it walks the mechanics of a
Tavus-class system, then what we built against them, then privacy, cost and production readiness.
The audience reads architecture documents for a living, so the deck's job is to be precise about
which claims are sourced and which are judgement.

**The rule this file exists under, and it constrains real content.** CLAUDE.md reserves three things
for the human: the build-vs-buy recommendation, the confirmed-vs-inferred tags, and the
what-would-change-my-mind thresholds. So every tagged claim here is **quoted from PROCESS.md §1–§2**,
which Prashanth authored, rather than composed here. Where the deck needs a claim §1 does not make,
it says `[HUMAN]` and names the gap instead of filling it.

**Three sections of §1 are unwritten scaffolds** -- §1.4 serving, §1.6 failure handling, §1.7
observability (an empty table) -- and nothing in the document flags them. Writing them would mean
inventing tagged architecture claims, which is exactly what is reserved. So the deck presents what
**our implementation** measurably does for those areas, which is fact rather than inference and needs
no tag, and marks the general-system claim as the author's to make. The gap is on its own slide.

    .venv/bin/python apps/api/scripts/make_assessment_deck.py --out nod-assessment.pptx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_deck import (
    ACCENT,
    BAD,
    GOOD,
    MEASURED,
    WARN,
    H,
    Inches,
    Presentation,
    W,
    audit,
    bullets,
    callout,
    content,
    flow,
    kpi,
    section,
    table,
    title_slide,
)

# Quoted verbatim from PROCESS.md §0, because a deck that restated the convention in its own words
# would be a second definition of the thing the assessment grades most closely.
TAGS = [
    ["[C] confirmed", "A primary source states this. Citation in §6."],
    ["[I] inferred", "Engineering judgement from observable behaviour, adjacent published work, or "
     "physical constraint. No source states it."],
    ["[U] unknown", "Could not be determined, and it matters. Stated rather than papered over."],
]


def build(out: Path) -> list[str]:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    title_slide(
        prs,
        "Engineering assessment · Head of Engineering",
        "How a real-time conversational\navatar system works",
        "Phase 1: the mechanics, with every claim tagged confirmed, inferred or unknown.\n"
        "Phase 2: what we built against them, and what it measured.",
    )

    s = content(prs, "How to read this deck", "Convention · quoted from PROCESS.md §0",
                notes="Open here, and do not skip it. The brief says knowing the difference between "
                      "confirmed and inferred is the point of Phase 1, so establishing the "
                      "vocabulary first buys you credit for every tag that follows. Say plainly "
                      "that the tags are quoted from the written document rather than invented for "
                      "the slides.")
    table(s, Inches(0.72), Inches(1.9), Inches(11.9), ["Tag", "Meaning"], TAGS,
          widths=[0.2, 0.8], size=12.5, row_h=0.5)
    callout(s, Inches(0.72), Inches(3.9), Inches(11.9), Inches(1.5),
            "Every tagged claim in this deck is quoted from PROCESS.md §1 and §2. None was composed "
            "for the slides. Where a claim is needed that the document does not make, the slide says "
            "so and names it as the author's to make — because a tag invented to fill a gap is worse "
            "than the gap.", tone=GOOD, label="Provenance")
    callout(s, Inches(0.72), Inches(5.6), Inches(11.9), Inches(1.2),
            "One thing said once, early: no talking-head model was integrated when the architecture "
            "document was written. It is now, on a Tesla T4, and every renderer figure in this deck "
            "is from that hardware. Where a number does not exist it says NOT YET MEASURED.",
            tone=ACCENT, label="Status")

    # ------------------------------------------------------------ Phase 1
    section(prs, "Phase 1", "The mechanics of a Tavus-class system",
            "Reverse-engineered from public documentation, published work, and physical constraint. "
            "No vendor access, by design.")

    s = content(prs, "It is not one model — it is a pipeline under orchestration",
                "§1.1 System overview",
                notes="The structural claim to land: the talking-head model is one bounded stage "
                      "near the end, and the interesting engineering is the orchestration around "
                      "it. If the audience takes one thing from Phase 1, it should be this — it is "
                      "also what makes the build-vs-buy conclusion later feel inevitable rather "
                      "than convenient.")
    flow(s, Inches(0.72), Inches(1.95), Inches(11.9), [
        ("Turn detection", "VAD + hysteresis. Streaming, local. NOT the vendor's endpointing — "
         "this decides when a turn is over."),
        ("STT", "Streaming. Mostly already finished by the time the silence window elapses."),
        ("LLM", "Streaming. Only time-to-first-token matters; total generation time does not."),
        ("TTS", "Streaming, per sentence. Sentence 2 synthesises while sentence 1 is playing."),
        ("Render", "Streaming. Needs only the first ~200 ms of audio to emit frame one."),
        ("Mixer + transport", "Constant cadence, owns timestamps, never stalls the track."),
    ], note_size=9)
    callout(s, Inches(0.72), Inches(3.5), Inches(11.9), Inches(1.6),
            "“The single most important structural claim in this document: the stages must overlap. "
            "Executed sequentially and to completion, the same components produce a multi-second "
            "turnaround with every one of them performing exactly to spec.” Executed as overlapping "
            "streams, the first frame emerges while the LLM is still writing.",
            tone=GOOD, label="§1.1, quoted")
    callout(s, Inches(0.72), Inches(5.3), Inches(11.9), Inches(1.5),
            "Every stage is streaming **except end-of-turn detection**, which is a deliberate wait — "
            f"{MEASURED['turn_detect']} ms here, and the single largest term in the measured budget. "
            "It is not a performance bug to optimise away; it is a conversational policy choice, and "
            "no amount of hardware changes it. Cutting it to 300 ms makes the system interrupt "
            "people mid-thought.", tone=ACCENT, label="The distinction that matters")

    s = content(prs, "Identity capture — data, not weights", "§1.2, with tags as written",
                notes="The serving consequence is the point of this slide: because identity is data "
                      "rather than per-person weights, one warm worker serves any persona and the "
                      "cost structure is per-GPU-second rather than per-persona. That is also the "
                      "claim most likely to differ at a vendor, and the tags say so.")
    table(s, Inches(0.72), Inches(1.9), Inches(11.9),
          ["Question", "Answer", "Tag"],
          [["Input to enrollment", "A short reference video or a single front-on image. MuseTalk "
            "operates on a 256×256 face region, so framing matters more than duration",
            "[C] MuseTalk · [I] vendors"],
           ["Artifact produced", "Per-frame VAE latents plus detection, parsing and bounding-box "
            "metadata. **No per-person model weights**", "[C] MuseTalk · [I] vendors"],
           ["Enrollment latency", "Dominated by face detection and VAE encoding. Was NOT YET "
            "MEASURED; now 126 s for a 550-frame reference on a T4", "[U] → measured since"],
           ["Reusable across sessions", "Yes — the artifact is a function of the reference clip "
            "alone, so it is computed once per persona and cached", "[I] a deduction"],
           ["Per-person GPU state", "Only cached latents and crop metadata. U-Net, VAE and audio "
            "encoder are shared across every persona on the worker", "[C] MuseTalk · [I] vendors"]],
          widths=[0.18, 0.58, 0.24], size=10.5, row_h=0.52)
    callout(s, Inches(0.72), Inches(5.15), Inches(11.9), Inches(1.65),
            "“Because identity is data rather than weights, one warm worker serves any persona.” "
            "That is what makes a warm pool economically possible at all — and it is the assumption "
            "a per-person-weights vendor design would break, which is why the vendor half of every "
            "row above is tagged [I] rather than [C].", tone=GOOD, label="The serving consequence")

    s = content(prs, "Which class of model survives the real-time constraint",
                "§1.3 The audio-to-video mechanism",
                notes="Walk the table top to bottom; the shape of the argument is that almost "
                      "everything dies, and it dies for different reasons — throughput, step count, "
                      "enrollment cost, realism. Then the narrow band that survives. This is the "
                      "slide that shows the space was actually researched rather than the first "
                      "GitHub repo taken.")
    table(s, Inches(0.72), Inches(1.9), Inches(11.9),
          ["Class", "Verdict", "Tag"],
          [["Full generative video diffusion (Sora-class)",
            "**Dies on throughput by orders of magnitude.** Solves a vastly larger problem than "
            "needed, and identity is not guaranteed stable across 20 minutes", "[I]"],
           ["Audio-conditioned latent diffusion, multi-step",
            "**Dies on step count.** N denoising steps multiplies per-frame cost by N against a "
            "~40 ms budget", "[I] — the budget is arithmetic, the conclusion judgement"],
           ["Motion-space diffusion + neural render (Ditto)",
            "**Survives, with an operational cost.** Diffuses in motion space, 50 → 10 steps, "
            "TensorRT-compiled. Engines are GPU-specific", "[C] — stated in the paper and repo"],
           ["Latent-space mouth inpainting, single-step (MuseTalk)",
            "**Survives, and is cheapest.** Borrows the SD v1.4 U-Net but is *not* a diffusion "
            "model — single-step inpainting. Claims 30fps+ on a V100", "[C] repo states all three"],
           ["3D rig driven by visemes",
            "**Survives on latency, dies on realism.** Trivially real-time and controllable, but "
            "looks animated rather than photographic", "[I]"],
           ["Gaussian splatting / NeRF per identity",
            "**Dies on enrollment, not inference.** Minutes to hours of per-person GPU work breaks "
            "the product shape and reintroduces per-person cost", "[I]"]],
          widths=[0.24, 0.61, 0.15], size=10, row_h=0.56)
    callout(s, Inches(0.72), Inches(5.5), Inches(11.9), Inches(1.3),
            "What survives is a narrow band: **single- or few-step generation, in a compact latent "
            "or motion space, over a small region of the frame.** Everything outside it fails on one "
            "of throughput, step count, enrollment cost or realism — and each for a different "
            "reason, which is why the table has six rows rather than a conclusion.",
            tone=GOOD, label="§1.3, the conclusion")

    s = content(prs, "The two sub-claims that separate a surface answer from a real one",
                "§1.3.1 – §1.3.2",
                notes="This is the slide that distinguishes having read a README from having read "
                      "the code. Most of the frame is replayed, not generated — which is why "
                      "identity survives and why the output is controllable. And temporal "
                      "consistency across chunk boundaries is a real problem with a real mechanism.")
    bullets(s, Inches(0.72), Inches(1.95), Inches(11.9), [
        ("Most of the frame is replayed, not generated. [C] for MuseTalk, [I] for vendors.",
         "The model repaints a mouth region inside frames the reference already supplied. That is "
         "why identity survives a 20-minute interview: it is not being re-imagined each frame, it "
         "is being reused. It is also why enrollment quality dominates output quality."),
        ("Temporal consistency comes from overlapping context, not from memory.",
         "Feature extraction either side of a window boundary overlaps, so the mouth does not jump "
         "between chunks. Our implementation keeps the audio tail as context for the next window — "
         "80 ms of it — for exactly this reason."),
        ("The honest name for the mechanism: audio-conditioned latent inpainting.",
         "`in_channels: 8` against `out_channels: 4` means two concatenated latents in, one out. "
         "`cross_attention_dim: 384` is exactly whisper-tiny's `d_model`, so audio enters where a "
         "text prompt would. `timestep=0`, no scheduler, no sampling loop."),
    ], size=13)

    # ------------------------------------------------------------ serving
    section(prs, "Phase 1b", "Serving, failure handling, observability",
            "Three areas where the architecture document is a scaffold. What follows is what our "
            "implementation measurably does — fact rather than inference, and no substitute for "
            "the general claim.")

    s = content(prs, "The gap, stated before the slides that work around it",
                "[HUMAN] §1.4, §1.6, §1.7",
                notes="Say this plainly and early. It is far better for them to hear it from you "
                      "than to find it. Three of the seven subsections of deliverable #1 are "
                      "outlines, and the brief explicitly asks for two of them — the latency budget "
                      "table, which exists, and where you would instrument observability, which is "
                      "an empty table. The next three slides give our measured implementation, "
                      "which is real but is not the same deliverable.")
    table(s, Inches(0.72), Inches(1.9), Inches(11.9),
          ["Section", "State", "What the brief asks for"],
          [["§1.1 System overview", "Written, tagged", "How the pipeline works end to end"],
           ["§1.2 Identity capture", "Written, tagged", "How a reference becomes reusable"],
           ["§1.3 Audio-to-video", "Written, tagged", "Which model class, and why"],
           ["§1.4 Serving architecture", "**Scaffold — 4 prompts**",
            "Warm pooling, transport, where the latency budget goes"],
           ["§1.5 Latency budget", "Written, measured", "A latency budget table"],
           ["§1.6 Failure and edge handling", "**Scaffold — 4 prompts**",
            "Interruption, silence, reconnect"],
           ["§1.7 Observability plan", "**Scaffold — empty table**",
            "Where you would instrument latency and failure telemetry"]],
          widths=[0.28, 0.24, 0.48], size=11, row_h=0.42,
          colors={(3, 1): BAD, (5, 1): BAD, (6, 1): BAD})
    callout(s, Inches(0.72), Inches(5.3), Inches(11.9), Inches(1.5),
            "Not filled in for the slides, deliberately. These sections need claims tagged confirmed "
            "or inferred about systems we have no access to, and a tag invented to complete a deck "
            "is worse than an acknowledged gap — it is the specific failure the brief's third graded "
            "standard is looking for. The three slides that follow are our own measured behaviour, "
            "which needs no tag because it is not a claim about anyone else.",
            tone=ACCENT, label="Why it is not simply written")

    s = content(prs, "Serving — what we measured, not what vendors do", "§1.4 · our implementation",
                notes="Every number here is ours and measured. The cold-start figure is the one that "
                      "makes the warm-pool argument concrete: 70–150 s of model loading cannot be "
                      "paid at conversation start, and we found out the hard way that loading per "
                      "session was exactly what the code did.")
    table(s, Inches(0.72), Inches(1.9), Inches(11.9),
          ["Concern", "What we measured on a T4"],
          [["Cold start, per process", "Models 22.5 s + first face 127 s. Warm-up moves both off "
            "the first candidate"],
           ["Cold start, per session — the bug",
            "`load()` filled an instance attribute while claiming to be per-process, so **every "
            "session reloaded 3.8 GB**. Audio at 6.2 s, first frame at 22.9 s"],
           ["First forward pass", f"{MEASURED['first_render_ms']} ms for five frames against "
            f"{MEASURED['render_ms']} ms steady — cuDNN algorithm choice, lazy allocators"],
           ["Session start, warm", MEASURED["session_start"]],
           ["Transport shipped", "WebSocket (13-byte binary header) and WebRTC via LiveKit, both "
            "behind one Protocol. The shortcut is documented, not hidden"],
           ["A/V sync, two publishers", f"{MEASURED['ws_drift_range']} — what two independent "
            "clocks produce"],
           ["A/V sync, one synchroniser",
            f"median {MEASURED['drift_median']}, stable to {MEASURED['drift_spread']}"],
           ["Concurrent sessions per GPU", "**NOT YET MEASURED.** Never tested above one"]],
          widths=[0.28, 0.72], size=10.5, row_h=0.46,
          colors={(6, 1): GOOD, (7, 1): BAD})

    s = content(prs, "Failure handling — implemented and measured", "§1.6 · our implementation",
                notes="Interruption is the one to dwell on because it is both the hardest and the "
                      "one we can demonstrate. The degradation ladder is the honest gap on this "
                      "slide — we do not have one, and saying so is better than describing graceful "
                      "degradation we never built.")
    table(s, Inches(0.72), Inches(1.9), Inches(11.9),
          ["Case", "Mechanism", "Measured"],
          [["Interruption (barge-in)",
            "Increment an epoch. In-flight work finishes; its frames die at the consumer as stale. "
            "No interruptible renderer needed",
            "0.6 ms server-side; survives a process boundary as an RPC"],
           ["Silence — nothing to show",
            "An idle loop built from the persona's own reference frames, so standing by is the same "
            "person rather than a placeholder", "0.40 mean / 0.69 peak motion vs 0.00 for a still"],
           ["Silence — nobody speaking", "A watchdog re-prompts after 12 s, and the re-prompt is "
            "recorded as a turn with an explicit `silent` flag", "Recorded; was invisible before"],
           ["Handover artifacts", "Cut to speech only on a mouth-closed frame, with a bounded "
            f"{'120'} ms wait and a `seam_forced` counter when it expires",
            f"{MEASURED['seams_forced']} seams forced across two turns"],
           ["Reconnect", "**[HUMAN] not designed.** Session state is in the store, so a reconnect "
            "could resume — untested and unspecified", "—"],
           ["Degradation ladder", "**[HUMAN] does not exist.** No resolution → fps → audio-only "
            "ladder. Naming one we have not built would be worse", "—"]],
          widths=[0.2, 0.5, 0.3], size=10.5, row_h=0.52,
          colors={(4, 1): WARN, (5, 1): WARN})

    s = content(prs, "Observability — the signals that actually exist", "§1.7 · our implementation",
                notes="The distinction worth making: telemetry here is not a side channel, it is the "
                      "only recorder of a turn. The stored transcript is built from the event stream "
                      "rather than from separate writes, so there is exactly one authority. The cost "
                      "of that design is that an unemitted event is an unrecorded turn — which is "
                      "precisely the silence re-prompt bug.")
    table(s, Inches(0.72), Inches(1.9), Inches(11.9),
          ["Signal", "Emitted at", "Answers"],
          [["latency: turn_detect", "End-of-turn decision", "Is the silence window right for this "
            "candidate?"],
           ["latency: llm_ttft", "First sentence", "Is the endpoint or the prompt the problem?"],
           ["latency: tts_first_audio", "First PCM chunk", "REST vs WebSocket, and voice choice"],
           ["latency: avatar_first_frame", "**Browser paint**, not socket write",
            "The only end-to-end number a candidate feels"],
           ["latency: perceived_total", "Turn start to first paint", "The headline SLO"],
           ["frames_repeated / frames_discarded", "Mixer",
            "Is the renderer keeping cadence, or is the queue draining?"],
           ["stale_artifact_dropped", "Epoch check", "Is cancellation actually working?"],
           ["heard (transcribed / silent)", "Turn open",
            "Broken STT vs a quiet candidate — different pages"],
           ["session_failure, labelled by cause", "Any unhandled turn error",
            "Which stage fails, not just that one did"],
           ["seam_forced", "Handover", "Is the idle clip's annotation too sparse?"]],
          widths=[0.27, 0.24, 0.49], size=10.5, row_h=0.4)
    callout(s, Inches(0.72), Inches(6.15), Inches(11.9), Inches(0.75),
            "**[HUMAN] What is missing is the alerting half**: §1.7's table has columns for type and "
            "threshold and they are empty. p50/p95/p99 aggregation and alert thresholds are a "
            "production design decision, not an implementation detail.",
            tone=ACCENT, label="The gap on this slide")

    # ------------------------------------------------------------ budget
    s = content(prs, "Where the latency actually goes", "§1.5 Latency budget · measured",
                notes="The conclusion this table exists to support is the backbone of the whole "
                      "submission: none of the three dominant terms is the renderer. Deliver that "
                      "line slowly and let it sit.")
    table(s, Inches(0.72), Inches(1.9), Inches(11.9),
          ["Stage", "Target", "Measured", "What moves it"],
          [["End-of-turn detection", "100–300 ms", f"{MEASURED['turn_detect']} ms (configured)",
            "Nothing in hardware. Speculative execution trades compute for latency"],
           ["Speech-to-text finalise", "50–150 ms", "~0 observed",
            "Streaming socket — hidden inside the silence window, not added to it"],
           ["LLM time-to-first-token", "200–500 ms", f"{MEASURED['llm_ttft']} ms",
            "Commercial, not architectural: a paid low-latency endpoint"],
           ["TTS time-to-first-audio", "100–300 ms", f"{MEASURED['tts_rest']} ms (REST)",
            f"Aura WebSocket measured {MEASURED['tts_ws']} ms — largest cheap win"],
           ["Avatar first frame", "50–150 ms", f"{MEASURED['start_lag']} after audio",
            "Render window plus the mixer's lead-in"],
           ["Encode + network", "50–150 ms", "20–25 ms",
            "Loopback, so a floor. A real network adds the client's jitter buffer"],
           ["**Perceived total**", "< 1,000 ms", MEASURED["turn_total"],
            "3–6× target, and the renderer is not why"]],
          widths=[0.24, 0.11, 0.21, 0.44], size=10.5, colors={(2, 2): BAD, (6, 2): BAD})
    callout(s, Inches(0.72), Inches(5.6), Inches(11.9), Inches(1.2),
            "Set the renderer to zero and roughly 2.6–5.7 s of a 2.7–5.8 s turn remains. **The part "
            "a vendor sells is the part that was never the problem.** The part that is — turn-taking, "
            "cancellation, history truncation, pipelining — is code no vendor API writes for you. "
            "“We need more GPU” is measurably the wrong diagnosis for this pipeline.",
            tone=GOOD, label="The finding everything else rests on")

    # ------------------------------------------------------------ Phase 2
    section(prs, "Phase 2", "What we built, and what it measured",
            "An open-source prototype of the same mechanics, on hardware we can name.")

    s = content(prs, "Model selection — licence eliminated the field first", "§2, as written",
                notes="The order of the criteria is the interesting part. Licence came before "
                      "performance, and it removed the best-known model in the category before any "
                      "fps figure was considered. That is a hiring-product constraint, not a "
                      "technical one, and it is the kind of judgement the brief is testing.")
    bullets(s, Inches(0.72), Inches(1.95), Inches(11.9), [
        ("The decisive criterion was licence, and it eliminated the field before performance.",
         "Wav2Lip is the best-known model in this category and its terms are unambiguous: personal, "
         "research and non-commercial only, with weights trained on LRS2 — “any form of commercial "
         "use is strictly prohibited”. For a candidate-interview product that ends the conversation; "
         "no fps figure rescues it."),
        ("MuseTalk: MIT code, weights permitting commercial use, single-step, published real time.",
         "And “it has not been made to run” at the time of writing — the spike failed in setup on a "
         "free-tier Colab T4 before touching the GPU. Both halves of that sentence were the honest "
         "answer then. It runs now, on a T4 we rented."),
        ("Ditto was the credible alternative and was rejected on operational cost.",
         "TensorRT engines are compiled per GPU architecture, and an ephemeral runtime hands out a "
         "different GPU each session. The decision is recorded with the threshold that would reverse "
         "it — [HUMAN] §2.3's “what would make me switch”."),
        ("The boundary is what makes the choice cheap to revisit.",
         "`TalkingHeadRenderer` is a Protocol with two implementations. Swapping the model is a "
         "change to one config value, which is why the memo can afford to be decisive."),
    ], size=12.5)

    s = content(prs, "What it does, end to end, verified by running it", "Phase 2 · prototype",
                notes="Lead with the fact that these were verified by driving the system rather than "
                      "by inspection — the brief's systems-depth axis asks for session mechanics, "
                      "not an offline render, and this is the slide that answers it.")
    kpi(s, Inches(0.72), Inches(1.9), Inches(3.75), MEASURED["tests_now"],
        "Tests, all GPU-free", "No GPU, no weights, no network — the boundary makes it possible",
        GOOD)
    kpi(s, Inches(4.79), Inches(1.9), Inches(3.75), MEASURED["fps_delivered"] + " fps",
        "Delivered live on a T4", "Against a configured target of 8. Capacity 12.8", GOOD)
    kpi(s, Inches(8.86), Inches(1.9), Inches(3.75), "2.0×",
        "Short of 25 fps real time", f"{MEASURED['render_ms']} ms/frame; VAE decode is 74%", WARN)
    bullets(s, Inches(0.72), Inches(3.6), Inches(11.9), [
        ("Session mechanics, not a model call.",
         "A seven-state machine, start/stop lifecycle, an idle loop built from the persona's own "
         "frames, and interruption that reacts in 0.6 ms server-side. Barge-in verified across a "
         "process boundary as an RPC."),
        ("Enrollment from a video or a photograph.",
         "A still holds one pose forever, so a photograph is animated by LivePortrait first — 500 "
         "frames, 20 s, 3.6% face-proportion deviation from the source. Enrollment is a background "
         "job: 202 in 0.018 s."),
        ("The product around it, because an interview is not a demo.",
         "Candidates with resumes that change what gets asked, a competency plan, async scoring "
         "whose quotes are re-checked against the transcript, egress recording, and a console for "
         "all of it."),
    ], size=12.5)

    # ------------------------------------------------------------ privacy
    section(prs, "Phase 2b", "Data privacy, residency and cost",
            "The premise of the brief was that candidate media leaves your infrastructure. Here is "
            "exactly how far we moved that, and what it would cost.")

    s = content(prs, "Where candidate data actually goes", "Data residency",
                notes="This is the slide that answers the brief's opening concern directly. Be "
                      "precise: the face never leaves, the conversation does. Do not round that up "
                      "to self-hosted.")
    table(s, Inches(0.72), Inches(1.9), Inches(11.9),
          ["Component", "Runs", "Candidate data that leaves"],
          [["Renderer (MuseTalk)", "your GPU", "nothing"],
           ["Enrollment (LivePortrait)", "your GPU / CPU", "nothing"],
           ["Voice cloning (Chatterbox)", "your GPU", "nothing"],
           ["Store (Postgres or JSON), resumes", "your host", "nothing"],
           ["Console, assistant, SFU, egress, Redis", "your host / Docker", "nothing"],
           ["Speech to text", "Deepgram", "**candidate audio**"],
           ["Text to speech", "Deepgram", "interviewer text only"],
           ["Language model", "OpenAI-compatible endpoint", "**transcript and prompt**"]],
          widths=[0.34, 0.24, 0.42], size=11.5,
          colors={(0, 2): GOOD, (1, 2): GOOD, (2, 2): GOOD, (3, 2): GOOD, (4, 2): GOOD,
                  (5, 2): BAD, (7, 2): BAD})
    callout(s, Inches(0.72), Inches(5.3), Inches(5.8), Inches(1.5),
            "**The face never leaves; the conversation does.** The LLM adapter speaks the OpenAI wire "
            "format precisely so Ollama, vLLM or LM Studio take over with a base-URL change and no "
            "code change.", tone=GOOD, label="What is already true")
    callout(s, Inches(6.82), Inches(5.3), Inches(5.8), Inches(1.5),
            "STT and TTS need real replacements, not configuration — Whisper for one, XTTS-v2 or "
            "F5-TTS for the other. Genuine work, unbuilt. And the store now holds real faces, "
            "voices and resumes with **no authentication in front of it**.",
            tone=BAD, label="What is not")

    s = content(prs, "Privacy posture, stated as a posture", "Security · biometric data",
                notes="The brief puts a formal security review out of scope, which is not the same "
                      "as pretending the exposure does not exist. This slide is the difference "
                      "between an omission and a decision — and the biometric framing is the one "
                      "that will matter to anyone in a regulated market.")
    bullets(s, Inches(0.72), Inches(1.95), Inches(11.9), [
        ("A face and a voice are biometric data, and in several jurisdictions a special category.",
         "Own consent, retention and deletion obligations. The store holds real people's reference "
         "media and resumes; that changes the calculus from untidy to serious, and it changed the "
         "moment the first real face was uploaded rather than at some later threshold."),
        ("Deletion actually deletes — this was a gap and is now closed.",
         "Removing a face removes the reference, the original upload, the thumbnail, the "
         "LivePortrait output directory and the ~1 GB prepared identity in the renderer's cache. "
         "Unlinking is confined to the media directory, so a record pointing elsewhere keeps its "
         "file and says so in the log."),
        ("Identity is attested, never verified — and every heading says so.",
         "There is no authentication, so nothing can prove who sat an interview. What is captured "
         "is a timestamped claim: typed name, expected name, consent, browser, timezone. `verified` "
         "is a field that is always false, present rather than omitted so nobody infers the absence "
         "of a check."),
        ("[HUMAN] Consent and retention are unbuilt.",
         "Who uploaded this face, on whose authority, and when is it deleted? There is no consent "
         "record and no retention policy — so deletion is triggered by an operator remembering, "
         "which is not a policy."),
    ], size=12.5)

    s = content(prs, "Cost — what we can state, and what we cannot", "Economics",
                notes="This is the most important honesty slide in the deck. We have measured "
                      "throughput. We have not measured concurrency, and cost per minute is "
                      "dominated by exactly that. Do not let anyone leave the room quoting a dollar "
                      "figure as measured — there deliberately is not one.")
    table(s, Inches(0.72), Inches(1.9), Inches(11.9),
          ["Input", "Status", "Value"],
          [["Render cost per frame, T4", "MEASURED", f"{MEASURED['render_ms']} ms"],
           ["Delivered fps, one session", "MEASURED", f"{MEASURED['fps_delivered']} fps"],
           ["GPU memory, one session + models", "MEASURED", "7.6 of 15 GB"],
           ["Enrollment, one persona", "MEASURED", "~126 s GPU, once, then cached"],
           ["**Concurrent sessions per GPU**", "**NOT YET MEASURED**",
            "Never tested above one. The term cost per minute depends on most"],
           ["Cloud GPU list price", "ASSUMPTION — not a quote",
            "PROCESS.md §4.2 assumes a T4-class instance at ~$0.35–0.50/hr"],
           ["Vendor per-minute price", "ASSUMPTION — no contract",
            "PROCESS.md §4.2 assumes ~$0.10–0.30/min at list"]],
          widths=[0.3, 0.22, 0.48], size=11,
          colors={(0, 1): GOOD, (1, 1): GOOD, (2, 1): GOOD, (3, 1): GOOD,
                  (4, 1): BAD, (5, 1): WARN, (6, 1): WARN})
    callout(s, Inches(0.72), Inches(5.4), Inches(11.9), Inches(1.4),
            "**There is deliberately no $/minute figure in this deck.** The arithmetic in §4.2 lands "
            "one to two orders of magnitude below the assumed vendor price — but it divides a "
            "measured GPU cost by an unmeasured concurrency number, and that number is the whole "
            "answer. One experiment closes it: run N concurrent sessions on one card and find where "
            "fps falls below target. Until then the direction is credible and the magnitude is not.",
            tone=ACCENT, label="Why the number is absent rather than estimated")

    # ------------------------------------------------------------ production
    section(prs, "Phase 2c", "Production readiness",
            "What would have to be true before a candidate who is not us sits one of these.")

    s = content(prs, "The gate list, in the order we would close it", "Production guidelines",
                notes="Order matters and is defensible: nothing above the line is optional, and "
                      "authentication is first because everything else on the list is secondary to "
                      "a store of biometric data with no door on it. Say that the ordering is a "
                      "judgement and invite them to challenge it.")
    table(s, Inches(0.72), Inches(1.9), Inches(11.9),
          ["#", "Gate", "Why it is where it is"],
          [["1", "Authentication and authorisation",
            "An operator login; a signed, expiring, single-use candidate link. Everything else is "
            "secondary to a store of faces and voices with no door on it"],
           ["2", "Consent record and retention policy",
            "Biometric data with no expiry is a liability that grows. Deletion works; the trigger "
            "for it does not exist"],
           ["3", "Self-hosted STT and TTS",
            "What makes “no candidate data leaves” true rather than aspirational. Real work, not "
            "configuration"],
           ["4", "Concurrency measured, then a warm pool",
            "Cost, capacity and the GPU-split decision all depend on one unmeasured number"],
           ["5", "A named degradation ladder",
            "Resolution, then fps, then audio-only with a static frame. “It would degrade "
            "gracefully” is not a design"],
           ["6", "Reconnect that resumes rather than restarts",
            "Session state is already in the store; the path is untested and unspecified"],
           ["7", "Rate limits and upload quotas",
            "200 MB per file with no per-caller limit makes filling a disk trivial"],
           ["8", "Audit trail beyond the assistant",
            "Who changed a rubric, who deleted a session. Currently only assistant writes are "
            "attributed"]],
          widths=[0.04, 0.28, 0.68], size=10.5, row_h=0.44)

    s = content(prs, "What would take it to real time", "Production · the 2.0×",
                notes="Be concrete about which levers are measured and which are not. The CPU "
                      "overlap is done and returned 1.59×; everything else here is unmeasured on our "
                      "hardware and the slide says so.")
    table(s, Inches(0.72), Inches(1.9), Inches(11.9),
          ["Lever", "Effect", "Evidence"],
          [["Overlap CPU with GPU",
            f"{MEASURED['render_ms_seq']} → {MEASURED['render_ms']} ms/frame", "**DONE — 1.59×**"],
           ["A faster VAE decoder (TAESD-class, TensorRT, fp8)",
            "Attacks 74% of the remaining frame", "NOT YET MEASURED — the largest lever"],
           ["A larger GPU (L4, A10, L40S)", "Shrinks the term that now dominates",
            "NOT YET MEASURED — nothing above a T4 tested"],
           ["Lower output resolution", "Does not touch VAE decode — fixed 256×256 work",
            "Measured: proposed and discarded for this reason"],
           ["Bigger batch", "No further gain; the curve is flat past 16", "Measured on the T4"],
           ["Renderer as its own participant",
            "A/V drift stable to 9 ms; makes the GPU split a deployment choice",
            "**Measured at a subscriber**"]],
          widths=[0.32, 0.34, 0.34], size=11,
          colors={(0, 2): GOOD, (5, 2): GOOD, (1, 2): WARN, (2, 2): WARN})
    callout(s, Inches(0.72), Inches(5.3), Inches(11.9), Inches(1.5),
            "One T4 serves a self-hosted face **or** a self-hosted voice, not both: "
            f"`avatar_first_frame` goes {MEASURED['voice_contention']} with the voice sidecar "
            "competing. Not memory — 7.6 of 15 GB — but compute contention, and lowering fps and "
            "batch size was tested and did not help. So it is a hardware or vendor decision, not a "
            "tuning exercise — and moving the renderer to its own participant turns it into a "
            "topology choice.", tone=ACCENT, label="The constraint that shapes deployment")

    # ------------------------------------------------------------ judgment
    section(prs, "Phase 2d", "Recommendation and migration",
            "These are the author's judgement. What follows is quoted, with the one input that has "
            "changed since it was written.")

    s = content(prs, "Build vs. buy", "Quoted from PROCESS.md §4",
                notes="Read the recommendation as written and do not extend it. Then be explicit "
                      "that one input changed: §4.2 says “I did not run a GPU”, and one has now "
                      "run. The measured render cost is far below the assumed vendor price, which "
                      "if anything strengthens the case — but re-deriving it is the author's call, "
                      "and saying so from a slide is the point of the third graded standard.")
    callout(s, Inches(0.72), Inches(1.9), Inches(11.9), Inches(1.5),
            "“Keep the vendor for the rendering stage. Build the orchestration layer in-house, "
            "starting now.” That is the hybrid, and the split is not a hedge — it falls directly out "
            "of what was measured.", tone=GOOD, label="The recommendation, as written")
    bullets(s, Inches(0.72), Inches(3.6), Inches(11.9), [
        ("The reasoning, unchanged: the model was never the bottleneck.",
         f"A full turn measures {MEASURED['turn_total']} against a sub-second target, and the three "
         "dominant terms are the end-of-turn policy, LLM time-to-first-token and TTS. Set the "
         "renderer to zero and 2.6–5.7 s remains."),
        ("What changed since it was written: a GPU ran.",
         "§4.2 opens “Every figure below is an assumption, not a quote. I have no vendor contract "
         "and did not run a GPU.” The second half is no longer true. Render cost per frame is "
         "measured and lands well below the assumed vendor per-minute price."),
        ("[HUMAN] The refresh this needs, and who owns it.",
         "Re-derive §4.2 with the measured render cost, run the concurrency experiment, and restate "
         "§4.4's thresholds. The recommendation may well survive intact — asserting that from a "
         "slide would be exactly the foregone conclusion the brief warns against."),
    ], size=12.5)

    s = content(prs, "Migration — cutover without a customer-facing regression",
                "Quoted from PROCESS.md §5",
                notes="The test in the brief is whether another senior engineer could execute this "
                      "without you in the room. Walk the phases and point at the preconditions — "
                      "the plan is gated on things being true before phase one starts, which is "
                      "what makes it executable rather than aspirational.")
    flow(s, Inches(0.72), Inches(1.95), Inches(11.9), [
        ("Preconditions", "Named and checkable before anything ships. A cutover gated on nothing is "
         "a hope."),
        ("Phase 1 · Shadow", "Render in parallel, serve the vendor. Compare quality and latency on "
         "real traffic, no candidate affected."),
        ("Phase 2 · Flagged", "Per-tenant flag, small cohorts first, abort criteria written down in "
         "advance."),
        ("Rollback", "One flag flip. The vendor path stays warm and paid for until decommission."),
        ("Decommission", "Only after a defined quiet period with no regression. Removing the "
         "fallback is last, not first."),
    ])
    bullets(s, Inches(0.72), Inches(3.5), Inches(11.9), [
        ("The boundary is what makes this cheap, and it is not a claim — it is demonstrated.",
         "The renderer is a Protocol with two implementations already, and a third — the LiveKit "
         "worker — was added without touching the orchestrator, the state machine or the turn "
         "policy. Shadow mode is one more implementation of an interface that exists."),
        ("Rollback is a flag, not a redeploy.",
         "Because session state lives in the orchestrator rather than in the renderer, and because "
         "the two renderers are interchangeable at the same seam. The paired-delivery experiment "
         "shipped behind exactly such a flag and was measured off again."),
    ], size=12.5)

    # ------------------------------------------------------------ close
    s = content(prs, "What is missing — the list we would want asked about",
                "Honest gaps",
                notes="Close on this rather than on the recommendation. Volunteering the gap list is "
                      "the argument for everything else in the deck being trustworthy, and it is "
                      "what the third graded standard is looking for. The first row is the one to "
                      "say slowly.")
    table(s, Inches(0.72), Inches(1.9), Inches(11.9),
          ["Gap", "Consequence", "Standing"],
          [["§1.4, §1.6, §1.7 are scaffolds",
            "Three subsections of deliverable #1 are outlines, including the observability table "
            "the brief asks for by name", "**Author's to write**"],
           ["No authentication anywhere",
            "The link is not a credential; the store holds real faces, voices and resumes",
            "Deferred deliberately"],
           ["Concurrency per GPU untested", "Cost per interview cannot be credibly stated",
            "One experiment away"],
           ["2.0× short of real time", f"{MEASURED['render_ms']} ms/frame; VAE decode is 74%",
            "Needs a decoder or a bigger card"],
           ["One T4 cannot host face + voice", f"{MEASURED['voice_contention']} with both",
            "Hardware or vendor decision"],
           ["Self-hosted STT/TTS unbuilt", "The conversation still leaves your infrastructure",
            "Real work, not configuration"],
           ["No output-quality comparison",
            "Our substituted landmark detector vs mmpose RTMPose is unmeasured for quality",
            "Known, unquantified"],
           ["Consent and retention", "Deletion works; nothing triggers it but memory",
            "Unbuilt"]],
          widths=[0.24, 0.5, 0.26], size=10.5, row_h=0.44,
          colors={(0, 2): BAD, (1, 2): WARN, (2, 2): WARN})

    s = content(prs, "Verify any of it", "Reproducibility",
                notes="End by handing over the commands. Every figure in this deck came from one of "
                      "these, and each writes a JSON file beside its console output so a number can "
                      "be traced rather than trusted.")
    table(s, Inches(0.72), Inches(1.9), Inches(11.9),
          ["Command", "Produces"],
          [["pytest -m \"not gpu\"", f"{MEASURED['tests_now']} tests: no GPU, no weights, no "
            "network"],
           ["python scripts/bench_renderer.py", "Throughput and per-stage tables, plus the "
            "sequential-vs-overlapped comparison"],
           ["python scripts/measure_lag.py --agent <id>",
            "Live delivery: fps, trailing gap, start lag, discards"],
           ["python scripts/avatar_worker.py --audio stream", "The renderer as a LiveKit "
            "participant, with A/V drift measured at a subscriber"],
           ["python scripts/avatar_sender.py --interrupt-after 3", "Barge-in across a process "
            "boundary, over RPC"],
           ["python scripts/smoke_session.py", "17 assertions over a real socket: barge-in, epoch "
            "drops, cadence, mic-driven turns"],
           ["curl localhost:8000/config", "Which implementation each boundary resolved to, warm-up "
            "state, schema problems"]],
          widths=[0.42, 0.58], size=11, row_h=0.44)
    callout(s, Inches(0.72), Inches(5.5), Inches(11.9), Inches(1.2),
            "MEASUREMENTS.md is the source of every number here, including a §9 that lists what is "
            "not measured and a record of two figures we published and later retracted because they "
            "had been measured on the wrong device. The retraction is in the document on purpose.",
            tone=GOOD, label="Where the numbers live")

    title_slide(prs, "Ends",
                "Clarity over cleverness.\nBoundaries over convenience.\nProof over opinion.",
                "Every figure came from a run on named hardware. Where a number does not exist, the "
                "slide says NOT YET MEASURED. Where a judgement is the author's, the slide says "
                "[HUMAN].")

    prs.save(str(out))
    print(f"wrote {out} — {len(prs.slides.__iter__.__self__._sldIdLst)} slides")
    # Same post-render check as the demo deck, and it has to be repeated here because this file
    # has its own `build` rather than calling that one. This deck draws from the same MEASURED
    # table, so it can acquire the same defect -- a plain string where an f-string was meant,
    # which no linter can see because the literal is valid and only the intent is wrong.
    return audit(prs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="nod-assessment.pptx")
    args = parser.parse_args()
    complaints = build(Path(args.out))
    for complaint in complaints:
        print(f"!! {complaint}")
    return 1 if complaints else 0


if __name__ == "__main__":
    raise SystemExit(main())
