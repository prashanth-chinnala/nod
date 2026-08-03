#!/usr/bin/env python3
"""
Build the product/architecture deck as a real .pptx.

**Why a script and not a hand-made file.** Every figure in this deck has to trace to a run. A
generator lets the numbers live in one dict, `MEASURED`, next to the source that produced them, so a
slide cannot quietly drift from MEASUREMENTS.md the way a hand-edited deck would. Re-run it after a
measurement changes and the deck changes with it.

**The two rules this file enforces, from CLAUDE.md.** No invented measurements: anything without a
run says `NOT YET MEASURED` and says why. And the judgment sections -- the build-vs-buy
recommendation, the confirmed-vs-inferred tags, the what-would-change-my-mind thresholds -- belong
to the human. Where this deck presents them it quotes PROCESS.md, which Prashanth authored, and
marks anything needing a refresh `[HUMAN]` rather than writing a new opinion.

    .venv/bin/python apps/api/scripts/make_deck.py --out nod-engineering-review.pptx
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

# -- palette ---------------------------------------------------------------------------------
# Chosen rather than defaulted: a deep slate-navy ground, one warm amber accent used sparingly,
# and neutrals biased very slightly blue so they read as part of the same family instead of as
# grey.
INK = RGBColor(0x14, 0x1B, 0x2D)
INK_MID = RGBColor(0x3A, 0x44, 0x5C)
INK_LOW = RGBColor(0x6B, 0x76, 0x8C)
PAPER = RGBColor(0xFA, 0xFB, 0xFD)
RULE = RGBColor(0xD8, 0xDD, 0xE6)
ACCENT = RGBColor(0xC9, 0x7B, 0x2A)
GOOD = RGBColor(0x2E, 0x6F, 0x4F)
WARN = RGBColor(0xB3, 0x5C, 0x1F)
BAD = RGBColor(0x9E, 0x2B, 0x2B)

BODY = "Arial"
W, H = Inches(13.333), Inches(7.5)


MEASURED = {
    # Every value here came from a run. The comment names the run or the document that records
    # it.
    "render_ms": "78.4",  # MEASUREMENTS.md §2, T4, batch 16, fp16, CPU overlapped
    "render_ms_seq": "124.7",  # same run, stages in sequence
    "render_speedup": "1.59×",
    "render_fps_capacity": "12.8",
    "fps_delivered": "8.3",  # §8b, live over WebSocket, target 8
    "fps_delivered_range": "8.2 – 8.9",
    "fps_before": "1.0 – 2.4",
    "gap_median": "−66 ms to +172 ms",  # two runs, medians disagreed in sign
    "gap_worst": "538 ms",
    "start_lag": "1,510 ms",
    "discards": "6 – 11",
    "discards_before": "33 – 81",
    "vae_ms": "57.8",
    "unet_ms": "12.3",
    "cpu_ms": "51.5",
    "gpu_ms": "70.1",
    "first_render_ms": "4,747",  # warm-up, 5 frames
    "models_ms": "22,524",
    "prepare_ms": "126,817",  # 500-frame reference
    "animate_ms": "213,112",  # LivePortrait, 500 frames
    "enroll_202_ms": "0.018",
    "tts_hosted_ms": "290",
    "tts_cloned_ms": "4,848",
    "llm_ttft": "1,645 / 2,942 / 4,724",
    "tts_rest": "869 / 956 / 889",
    "tts_ws": "351 – 361",
    "turn_total": "2.7 – 5.8 s",
    "turn_detect": "700",
    "scorer": "8 ms to queue, 6,093 ms of work",
    "recording": "7.0 MB, 1 m 37 s, H.264 + AAC",
    "webrtc_first": "4,296 ms perceived / 4,221 ms paint",
    "tests": "746",
    "session_start": "1.5 – 3.8 s",
    "voice_contention": "3 s → 28 s",
}


def add(prs: Presentation) -> object:
    return prs.slides.add_slide(prs.slide_layouts[6])


def rect(slide, x, y, w, h, fill=None, line=None, line_w=1.0):
    from pptx.enum.shapes import MSO_SHAPE

    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    if fill is None:
        shape.fill.background()
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(line_w)
    shape.shadow.inherit = False
    return shape


def text(
    slide, x, y, w, h, runs, size=14, color=INK, bold=False, align=PP_ALIGN.LEFT, spacing=1.15
):
    """`runs` is a string or a list of (text, {overrides}) tuples, one paragraph per list item."""
    box = slide.shapes.add_textbox(x, y, w, h)
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    items = [runs] if isinstance(runs, str) else runs
    for index, item in enumerate(items):
        body, over = (item, {}) if isinstance(item, str) else item
        para = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        para.alignment = over.get("align", align)
        para.line_spacing = over.get("spacing", spacing)
        if over.get("space_before"):
            para.space_before = Pt(over["space_before"])
        run = para.add_run()
        run.text = body
        font = run.font
        font.name = over.get("font", BODY)
        font.size = Pt(over.get("size", size))
        font.bold = over.get("bold", bold)
        font.color.rgb = over.get("color", color)
    return box


def title_slide(prs, kicker, headline, sub):
    slide = add(prs)
    rect(slide, 0, 0, W, H, fill=INK)
    rect(slide, Inches(0.9), Inches(2.35), Inches(1.1), Pt(4), fill=ACCENT)
    text(slide, Inches(0.9), Inches(1.75), Inches(11), Inches(0.4),
         kicker.upper(), size=12, color=ACCENT, bold=True)
    text(slide, Inches(0.9), Inches(2.7), Inches(11.2), Inches(1.6),
         headline, size=40, color=PAPER, bold=True, spacing=1.0)
    text(slide, Inches(0.9), Inches(4.5), Inches(10), Inches(1.2),
         sub, size=15, color=RGBColor(0xA8, 0xB2, 0xC6), spacing=1.3)
    return slide


def section(prs, number, headline, sub=""):
    slide = add(prs)
    rect(slide, 0, 0, W, H, fill=INK)
    text(slide, Inches(0.9), Inches(2.9), Inches(2), Inches(0.8),
         number, size=13, color=ACCENT, bold=True)
    text(slide, Inches(0.9), Inches(3.3), Inches(11), Inches(1.2),
         headline, size=34, color=PAPER, bold=True, spacing=1.0)
    if sub:
        text(slide, Inches(0.9), Inches(4.5), Inches(10.5), Inches(1),
             sub, size=14, color=RGBColor(0xA8, 0xB2, 0xC6), spacing=1.3)
    return slide


def content(prs, heading, kicker="", notes=""):
    slide = add(prs)
    rect(slide, 0, 0, W, H, fill=PAPER)
    top = Inches(0.62)
    if kicker:
        text(slide, Inches(0.72), top, Inches(11.5), Inches(0.3),
             kicker.upper(), size=10.5, color=ACCENT, bold=True)
        top = Inches(0.95)
    text(slide, Inches(0.72), top, Inches(11.9), Inches(0.6),
         heading, size=25, color=INK, bold=True, spacing=1.0)
    rect(slide, Inches(0.72), Inches(1.62), Inches(11.9), Pt(1.1), fill=RULE)
    if notes:
        slide.notes_slide.notes_text_frame.text = notes
    return slide


def table(slide, x, y, w, headers, rows, widths=None, size=11.5, header_size=10,
          row_h=0.34, colors=None):
    """
    A table drawn from rectangles rather than a pptx table.

    Deliberate: pptx tables carry a theme style that overrides per-cell colour in ways that
    differ between PowerPoint, Keynote and Google Slides, and a deck whose whole point is
    legibility of numbers cannot have its emphasis silently dropped by the viewer's app.
    """
    widths = widths or [1.0 / len(headers)] * len(headers)
    total = float(w)
    xs, running = [], float(x)
    for fraction in widths:
        xs.append(running)
        running += total * fraction

    text(slide, Emu(int(x)), Emu(int(y)), Emu(int(w)), Inches(0.25), "", size=1)
    for index, head in enumerate(headers):
        text(slide, Emu(int(xs[index])), Emu(int(y)), Emu(int(total * widths[index] - 60000)),
             Inches(0.26), head.upper(), size=header_size, color=INK_LOW, bold=True)
    rect(slide, Emu(int(x)), Emu(int(y + Inches(0.3))), Emu(int(w)), Pt(0.9), fill=RULE)

    cursor = y + Inches(0.42)
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            colour = INK if c == 0 else INK_MID
            bold = c == 0
            if colors and (r, c) in colors:
                colour = colors[(r, c)]
                bold = True
            text(slide, Emu(int(xs[c])), Emu(int(cursor)),
                 Emu(int(total * widths[c] - 60000)), Inches(row_h),
                 cell, size=size, color=colour, bold=bold, spacing=1.05)
        cursor += Inches(row_h)
        if r < len(rows) - 1:
            rect(slide, Emu(int(x)), Emu(int(cursor - Inches(0.06))), Emu(int(w)),
                 Pt(0.6), fill=RGBColor(0xEC, 0xEF, 0xF4))
    return cursor


def bullets(slide, x, y, w, items, size=13, gap=0.1):
    cursor = float(y)
    for item in items:
        head, body = (item, "") if isinstance(item, str) else item
        rect(slide, Emu(int(x)), Emu(int(cursor + Inches(0.07))), Pt(3.2), Pt(3.2), fill=ACCENT)
        runs = [(head, {"bold": True, "size": size, "color": INK})]
        if body:
            runs.append((body, {"size": size - 1, "color": INK_MID, "spacing": 1.25}))
        text(slide, Emu(int(x + Inches(0.22))), Emu(int(cursor)), Emu(int(w - Inches(0.22))),
             Inches(0.3), runs)
        lines = 1 + (len(body) // 95 if body else 0)
        cursor += Inches(0.28 + 0.19 * lines + gap)
    return cursor


def kpi(slide, x, y, w, value, label, note="", colour=INK):
    rect(slide, Emu(int(x)), Emu(int(y)), Emu(int(w)), Inches(1.42),
         fill=RGBColor(0xFF, 0xFF, 0xFF), line=RULE)
    rect(slide, Emu(int(x)), Emu(int(y)), Pt(3), Inches(1.42), fill=colour)
    text(slide, Emu(int(x + Inches(0.22))), Emu(int(y + Inches(0.16))),
         Emu(int(w - Inches(0.4))), Inches(0.5), value, size=25, color=colour, bold=True)
    text(slide, Emu(int(x + Inches(0.22))), Emu(int(y + Inches(0.72))),
         Emu(int(w - Inches(0.4))), Inches(0.3), label, size=10.5, color=INK, bold=True)
    if note:
        text(slide, Emu(int(x + Inches(0.22))), Emu(int(y + Inches(0.98))),
             Emu(int(w - Inches(0.4))), Inches(0.4), note, size=9.5, color=INK_LOW, spacing=1.15)


def callout(slide, x, y, w, h, body, tone=ACCENT, label=""):
    rect(slide, Emu(int(x)), Emu(int(y)), Emu(int(w)), Emu(int(h)),
         fill=RGBColor(0xF4, 0xF1, 0xEA) if tone == ACCENT else RGBColor(0xF2, 0xF6, 0xF3))
    rect(slide, Emu(int(x)), Emu(int(y)), Pt(3), Emu(int(h)), fill=tone)
    top = y + Inches(0.16)
    if label:
        text(slide, Emu(int(x + Inches(0.24))), Emu(int(top)), Emu(int(w - Inches(0.5))),
             Inches(0.25), label.upper(), size=9.5, color=tone, bold=True)
        top += Inches(0.28)
    text(slide, Emu(int(x + Inches(0.24))), Emu(int(top)), Emu(int(w - Inches(0.5))),
         Emu(int(h)) - Inches(0.3), body, size=12, color=INK_MID, spacing=1.3)


def flow(slide, x, y, w, steps, note_size=9.5):
    """A horizontal pipeline. Each step is (title, detail)."""
    n = len(steps)
    gap = Inches(0.16)
    box_w = (Emu(int(w)) - gap * (n - 1)) / n
    for index, (head, detail) in enumerate(steps):
        left = Emu(int(x)) + (box_w + gap) * index
        rect(slide, Emu(int(left)), Emu(int(y)), Emu(int(box_w)), Inches(1.15),
             fill=RGBColor(0xFF, 0xFF, 0xFF), line=RULE)
        rect(slide, Emu(int(left)), Emu(int(y)), Emu(int(box_w)), Pt(2.4), fill=INK)
        text(slide, Emu(int(left + Inches(0.14))), Emu(int(y + Inches(0.18))),
             Emu(int(box_w - Inches(0.28))), Inches(0.3), head, size=11.5, color=INK, bold=True)
        text(slide, Emu(int(left + Inches(0.14))), Emu(int(y + Inches(0.5))),
             Emu(int(box_w - Inches(0.28))), Inches(0.6), detail,
             size=note_size, color=INK_LOW, spacing=1.15)
        if index < n - 1:
            text(slide, Emu(int(left + box_w + Inches(0.01))), Emu(int(y + Inches(0.42))),
                 Emu(int(gap)), Inches(0.3), "›", size=15, color=ACCENT, bold=True,
                 align=PP_ALIGN.CENTER)


def build(out: Path) -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # ---------------------------------------------------------------- 1. title
    title_slide(
        prs,
        "Engineering review · confidential",
        "A real-time conversational\navatar interview system",
        "Built in-house, end to end: identity capture, audio-driven face rendering, session "
        "mechanics, scoring.\nEvery number in this deck came from a run on named hardware. "
        "Where a number does not exist, the slide says so.",
    )

    # ---------------------------------------------------------------- 2. exec summary
    s = content(prs, "Where the system stands", "Executive summary",
                notes="Open on the honest frame: this is a working product that is not yet fast "
                      "enough, and we can say exactly how much not-enough and why. The three "
                      "KPIs are all from runs on a Tesla T4. 8.3 fps is against a configured "
                      "target of 8, not against 25 — say that out loud before anyone asks.")
    kpi(s, Inches(0.72), Inches(1.9), Inches(3.75), MEASURED["fps_delivered"] + " fps",
        "Delivered to the client, live", "Against a configured target of 8. Capacity is "
        f"{MEASURED['render_fps_capacity']} fps.", GOOD)
    kpi(s, Inches(4.79), Inches(1.9), Inches(3.75), MEASURED["gap_median"],
        "Audio-to-video trailing gap", "Median of two runs; they disagree in sign. "
        f"Worst single turn {MEASURED['gap_worst']}.", GOOD)
    kpi(s, Inches(8.86), Inches(1.9), Inches(3.75), "2.0×",
        "Short of 25 fps real time", f"{MEASURED['render_ms']} ms/frame against a 40 ms budget. "
        "VAE decode is 74% of it.", WARN)
    bullets(s, Inches(0.72), Inches(3.6), Inches(11.9), [
        ("Working and verified by driving it, not by inspection.",
         "Two-way conversation with barge-in, a real face from an uploaded video or an animated "
         "photograph, competency-planned questions, async scoring, a report view, egress "
         "recording, and a screen-aware operator assistant."),
        ("The renderer was never the bottleneck — and we measured that twice.",
         f"A full turn is {MEASURED['turn_total']}; set the renderer to zero and ~2.6–5.7 s "
         "remains. Separately, delivered fps was 1.4 while the card could do 12.8: the fault was "
         "our orchestration, not the model."),
        ("Two things are open, and one of them is a business decision.",
         "One T4 serves a self-hosted face or a self-hosted voice, not both. And there is no "
         "authentication anywhere — a stated development posture, not an oversight."),
    ])

    # ---------------------------------------------------------------- 3. alignment
    section(prs, "01", "Does this answer the brief?",
            "The assessment asks for five deliverables. Four exist and are current; one needs a "
            "refresh because the facts under it changed.")

    s = content(prs, "Assessment deliverables, mapped", "Alignment",
                notes="Be direct about the one amber row. PROCESS.md was written in ~1.5 days "
                      "before any GPU ran. Its architecture research and model memo still stand. "
                      "Its measured sections were superseded, and the build-vs-buy cost model "
                      "explicitly says 'I did not run a GPU' — that input now exists, so the memo "
                      "deserves a refresh. That is a judgment call for the author, not something "
                      "to patch silently.")
    table(s, Inches(0.72), Inches(1.9), Inches(11.9),
          ["Deliverable", "Where it lives", "State"],
          [["1 · Architecture document", "PROCESS.md §1 — [C]/[I]/[U] tags, latency budget, "
            "observability plan", "Current"],
           ["2 · Model-selection memo", "PROCESS.md §2 — criteria, weights, candidates, the "
            "argument against the pick", "Current"],
           ["3 · Working prototype + process doc", "This repo; ARCHITECTURE / MODELS / "
            "MEASUREMENTS / OPERATIONS / SECURITY", "Current"],
           ["4 · Build-vs-buy memo", "PROCESS.md §4 — recommendation, cost model, "
            "what-would-change-my-mind", "Needs refresh"],
           ["5 · Migration plan", "PROCESS.md §5 — shadow mode, flagged rollout, rollback, "
            "decommission", "Current"],
           ["CI workflow", ".github/workflows/ci.yml — lint, types, "
            f"{MEASURED['tests']} GPU-free tests", "Current"],
           ["Incremental commit history", "88 commits on the product branch beyond the "
            "submission tag", "Current"]],
          widths=[0.29, 0.53, 0.18],
          colors={(0, 2): GOOD, (1, 2): GOOD, (2, 2): GOOD, (3, 2): WARN, (4, 2): GOOD,
                  (5, 2): GOOD, (6, 2): GOOD})
    callout(s, Inches(0.72), Inches(5.5), Inches(11.9), Inches(1.3),
            "§4.2 of the build-vs-buy memo opens: “Every figure below is an assumption, not a "
            "quote. I have no vendor contract and did not run a GPU.” The second half is no "
            "longer true — a T4 ran, and render throughput is now measured. The recommendation "
            "may well survive the new input, but re-deriving it is the author's call.",
            tone=ACCENT, label="[HUMAN] the one gap worth naming first")

    s = content(prs, "Graded standards, and where we stand", "Alignment",
                notes="Standard 6 is the one to dwell on: documentation discipline means the "
                      "process doc and the prototype describe the same reality. PROCESS.md now "
                      "carries a banner saying it is historical and pointing at the current docs, "
                      "plus a 'status today' row with the measured figures. That is the fix — "
                      "not rewriting history, but making the divergence impossible to miss.")
    table(s, Inches(0.72), Inches(1.9), Inches(11.9),
          ["#", "Standard", "Evidence in this system"],
          [["1", "Built to last", "58 runtime modules, typed and linted; docs written to inform a "
            "decision, not to demo"],
           ["2", "Deterministic governance of ML", "Seven-state machine, epoch cancellation, "
            "idle loop — all deterministic code around a swappable model"],
           ["3", "Honest risk communication", "[C]/[I]/[U] tags; two of our own published figures "
            "retracted as wrong after re-measurement"],
           ["4", "Contracts first-class", "contracts.py imports nothing; boundary enforced by a "
            "test that inspects sys.modules"],
           ["5", "Mandatory observability", "Telemetry is the only turn recorder; latency stages "
            "named and emitted per turn"],
           ["6", "Documentation discipline", "PROCESS.md flagged historical with pointers + a "
            "measured status row; corrections logged, not edited away"],
           ["7", "Boring technology", "MuseTalk + LivePortrait + Deepgram + FastAPI. No novel "
            "model, no from-scratch training"]],
          widths=[0.04, 0.24, 0.72], size=11)

    # ---------------------------------------------------------------- 4. product flow
    section(prs, "02", "How the product works",
            "Three flows: enrolling a persona, running a live interview, and everything that "
            "happens after the candidate leaves.")

    s = content(prs, "Flow 1 — Enrolling a persona", "App flow",
                notes="Two entry points, one output. A video gives the best result because "
                      "MuseTalk repaints the mouth of frames it is given — with one still frame "
                      "the head never moves. That is why the photo path animates first. Both "
                      "paths converge on the same prepared identity, so nothing downstream knows "
                      "which was used. Enrollment is asynchronous, not offline: the operator does "
                      "it in our console and it returns immediately.")
    flow(s, Inches(0.72), Inches(1.95), Inches(11.9), [
        ("Upload", "Video, or a photo. Probed for codec, duration, face presence; rejected with a "
         "reason if unusable."),
        ("Animate (photo only)", "LivePortrait transfers motion from a driving clip so a still "
         "can blink and turn."),
        ("Detect + crop", "face_alignment finds 68 landmarks per frame; the mouth region is cut "
         "on upstream's exact arithmetic."),
        ("Encode to latents", "The VAE turns each 256×256 crop into 32×32×4. Masks and blend "
         "boxes are precomputed."),
        ("Prepared identity", "Cycled frames, latents, masks, plus an idle loop picked from the "
         "quietest quarter of frames."),
    ])
    table(s, Inches(0.72), Inches(3.5), Inches(11.9),
          ["Step", "Measured on a T4", "Note"],
          [["POST /faces/{id}/prepare returns", f"{MEASURED['enroll_202_ms']} s (HTTP 202)",
            "A worker thread does the work; the row is claimed with a timestamp"],
           ["Photo → 20 s moving reference", f"{MEASURED['animate_ms']} ms",
            "LivePortrait, 500 frames, CPU ONNX providers"],
           ["Reference → prepared identity", f"{MEASURED['prepare_ms']} ms",
            "500 frames. Cached process-wide, so a session pays nothing"],
           ["Voice clone from a recording", f"{MEASURED['tts_cloned_ms']} ms to first audio",
            "Chatterbox in its own process; see the GPU contention slide"]],
          widths=[0.29, 0.22, 0.49], size=11)
    callout(s, Inches(0.72), Inches(5.75), Inches(11.9), Inches(1.05),
            "Identity preservation is LivePortrait's design, not a hope: it transfers motion via "
            "implicit keypoints and does not swap faces. Measured at 3.6% face-proportion "
            "deviation from the source photograph. A model that invented motion would be free to "
            "invent a different person — worse than a frozen head.", tone=GOOD, label="Why this model")

    s = content(prs, "Flow 2 — A live interview turn", "App flow",
                notes="This is the slide to walk slowly. The key insight is that audio and video "
                      "are produced by different clocks and reconciled at the mixer. Note that "
                      "the LLM streams sentences, not tokens — TTS needs a whole sentence for "
                      "prosody, so sentence assembly is where the pipeline naturally chunks.")
    flow(s, Inches(0.72), Inches(1.95), Inches(11.9), [
        ("Mic in", "16 kHz PCM over the socket. Energy VAD plus a 700 ms silence window decides "
         "the turn is over."),
        ("Transcribe", "Deepgram nova-3 on a persistent socket, so the transcript is final by the "
         "time the window elapses."),
        ("Plan + think", "The competency plan picks what to probe; the LLM streams back "
         "sentences, not tokens."),
        ("Speak", "Deepgram Aura 2 per sentence. Real-time factor below 1.0 keeps generation "
         "ahead of playback."),
        ("Render + mix", "The same audio drives the face. The mixer emits at cadence and "
         "restamps every frame."),
    ])
    table(s, Inches(0.72), Inches(3.5), Inches(11.9),
          ["Mechanism", "What it does", "Why it is built that way"],
          [["Epoch cancellation", "Barge-in increments an integer; in-flight frames die at the "
            "consumer", "Reaction is one write. Wasted work is bounded by one render window"],
           ["Idle loop", "The persona's own frames play when nobody is speaking",
            "Built from the reference, so standing-by is the same person, not a grey placeholder"],
           ["Acknowledged-audio truncation", "History is cut to what the browser reports it "
            "played", "Otherwise an interrupted turn enters history as if fully heard"],
           ["Clean-seam exit", "The idle loop only cuts to speech on a mouth-closed frame",
            "A cut from mid-vowel to mid-vowel is visible; there is a bounded wait for a seam"]],
          widths=[0.2, 0.35, 0.45], size=11)

    s = content(prs, "Flow 3 — After the candidate leaves", "App flow",
                notes="Scoring is async because it takes ~6 seconds of model work and nobody "
                      "should hold a socket for it. The 8 ms figure is the API's own latency to "
                      "accept the job. The report reads only what telemetry recorded, which is "
                      "why the silence-re-prompt bug mattered: an unrecorded turn is invisible to "
                      "every downstream consumer.")
    flow(s, Inches(0.72), Inches(1.95), Inches(11.9), [
        ("Turns persisted", "Built from the telemetry stream, not from separate write calls — one "
         "authority, no disagreement."),
        ("Score", "Queued in 8 ms; ~6 s of model work off the request path. Quotes are checked "
         "against the transcript."),
        ("Report", "Per-competency scores, coverage, the transcript, and every latency stage the "
         "turn recorded."),
        ("Recording", "LiveKit egress writes a real H.264/AAC MP4 — a recorder the SFU binary "
         "does not include."),
    ])
    table(s, Inches(0.72), Inches(3.5), Inches(11.9),
          ["Measured", "Value", "Source"],
          [["Scorer: accept vs. work", MEASURED["scorer"], "Live run"],
           ["Egress recording", MEASURED["recording"], "Live run, file on disk"],
           ["WebRTC first frame", MEASURED["webrtc_first"], "requestVideoFrameCallback, not the "
            "decoder — the 74.9 ms paint tail was invisible before"],
           ["Session start, warm", MEASURED["session_start"], "T4, models and face already warm"]],
          widths=[0.24, 0.24, 0.52], size=11)

    # ---------------------------------------------------------------- 5. internals
    section(prs, "03", "Internal mechanism",
            "Four decisions carry most of the system's behaviour. Each was made for a reason we "
            "can state, and each has a test that would fail if it were dropped.")

    s = content(prs, "The four ideas that matter", "Internals",
                notes="If you remember one thing: contracts.py imports nothing from the package, "
                      "and that is enforced by a test that inspects sys.modules after import "
                      "rather than trusting the source. That single property is what lets 746 "
                      "tests run with no GPU, no weights and no network.")
    bullets(s, Inches(0.72), Inches(1.95), Inches(11.9), [
        ("Everything is a Protocol in contracts.py, which imports nothing.",
         "Renderer, transport, LLM, TTS, transcriber. It is the only module every layer may depend "
         "on, so no layer depends on another. Enforced by a test that inspects sys.modules after "
         "import — that is why the whole suite runs GPU-free, weight-free and offline."),
        ("Cancellation is an integer.",
         "Interrupting increments an epoch. In-flight GPU work still finishes and its frames die "
         "at the consumer because their epoch is stale. No interruptible renderer required."),
        ("History truncates to what was heard, not what was sent.",
         "The browser reports played milliseconds from Web Audio's own clock. Without it, an "
         "interrupted turn enters history as though fully delivered and the next question refers "
         "to a sentence nobody heard."),
        ("Composition, not configuration.",
         "Knowledge, pronunciation, guardrails and the competency plan each wrap a sentence "
         "stream: with_plan(with_guardrail(with_knowledge(llm))). The orchestrator does not know "
         "they exist."),
    ], size=13.5)
    callout(s, Inches(0.72), Inches(5.75), Inches(11.9), Inches(1.05),
            "The boundary is a graded requirement, not a style preference — so it is checked "
            "mechanically. tests/test_boundaries.py asserts no torch, CUDA or renderer "
            "implementation reaches the orchestrator, mixer or state machine; "
            "tests/test_renderer_contract.py builds every renderer from every option combination "
            "the server can pass.", tone=GOOD, label="How the boundary is kept honest")

    s = content(prs, "The session state machine", "Internals",
                notes="Seven states. The table in state.py maps each to a frame source, which is "
                      "why 'which state shows which picture' is a table a test can walk rather "
                      "than behaviour scattered through the pipeline. CANCELLING exists so a "
                      "barge-in has somewhere to land that is not LISTENING — the flush has to "
                      "happen before new audio is accepted.")
    table(s, Inches(0.72), Inches(1.9), Inches(11.9),
          ["State", "Frame source", "Leaves when"],
          [["INITIALIZING", "none", "The identity is prepared and the idle loop is installed"],
           ["IDLE", "idle loop", "Speech starts, or the silence watchdog fires at 12 s"],
           ["LISTENING", "idle loop", "700 ms of silence ends the turn"],
           ["THINKING", "idle loop", "Enough rendered frames are buffered and the seam is clean"],
           ["SPEAKING", "rendered frames", "The turn's audio is exhausted, or a barge-in arrives"],
           ["CANCELLING", "idle loop", "The client's audio buffer is flushed and the epoch bumped"],
           ["CLOSED", "none", "Terminal"]],
          widths=[0.17, 0.18, 0.65], size=11.5)
    callout(s, Inches(0.72), Inches(5.0), Inches(5.8), Inches(1.8),
            "The silence watchdog re-prompts after 12 s. Until recently that turn was generated, "
            "spoken, heard by the candidate — and stored nowhere, because the server builds turns "
            "from the telemetry stream and the watchdog emitted no 'heard' event. The transcript "
            "jumped between answers with no sign anything had happened.",
            tone=ACCENT, label="A bug this design made findable")
    callout(s, Inches(6.82), Inches(5.0), Inches(5.8), Inches(1.8),
            "Fixed with an explicit 'silent' column rather than an inferred empty transcript — "
            "because an empty transcript already means something urgent and different: speech was "
            "detected and the transcriber returned nothing. Collapsing the two would make a quiet "
            "candidate and a broken STT key identical in the record.",
            tone=GOOD, label="Why it is a column, not an inference")

    s = content(prs, "What the renderer actually does", "Internals",
                notes="This is the slide that shows we understand the model rather than having "
                      "cloned a README. MuseTalk is not a diffusion model in operation: "
                      "timestep=0, no scheduler, no sampling loop, one forward pass. in_channels "
                      "8 vs out_channels 4 means it eats two concatenated latents. "
                      "cross_attention_dim 384 is exactly whisper-tiny's d_model, so audio enters "
                      "where a text prompt would. The honest name is audio-conditioned latent "
                      "inpainting — it never synthesises a person, it repaints a mouth.")
    bullets(s, Inches(0.72), Inches(1.95), Inches(11.9), [
        ("It is not a diffusion model in operation.",
         "timestep=0, no scheduler, no sampling loop — one forward pass per frame. That is the "
         "whole reason it approaches real time, and it is why fps scales with batch rather than "
         "with step count."),
        ("in_channels 8, out_channels 4 — it inpaints, it does not generate.",
         "Two concatenated latents go in: the masked lower face, and an intact reference. One "
         "comes out. It never synthesises a person; it repaints the mouth of frames you supplied."),
        ("cross_attention_dim 384 is exactly whisper-tiny's d_model.",
         "Audio enters where a text prompt would in an image model. That is the conditioning "
         "mechanism, and it is why the audio encoder cannot be swapped casually."),
        ("The honest description: audio-conditioned latent inpainting.",
         "Stating it this way sets the right expectation with a customer. Identity comes from the "
         "reference, not from the model — which is also why enrollment quality dominates output "
         "quality."),
    ], size=13.5)

    # ---------------------------------------------------------------- 6. metrics
    section(prs, "04", "Measured performance",
            "Hardware: NVIDIA Tesla T4, 15 GB, 4 vCPU. Every figure below came from a run. "
            "Nothing here is a target presented as a result.")

    s = content(prs, "Renderer throughput, and where the time goes", "Metrics · T4, batch 16, fp16",
                notes="Two things to land. First, the 1.59× came from overlapping the CPU half of "
                      "a frame with the GPU half — they are different hardware and were taking "
                      "turns. Second, the stage table no longer sums to the frame cost, and that "
                      "is the point: 78.4 against a GPU-only floor of 70.1 means the CPU work is "
                      "genuinely hidden, not merely moved.")
    table(s, Inches(0.72), Inches(1.9), Inches(6.4),
          ["Stage", "Hardware", "ms/frame"],
          [["Positional encoding", "GPU", "0.0"],
           ["U-Net forward", "GPU", MEASURED["unet_ms"]],
           ["VAE decode", "GPU", MEASURED["vae_ms"]],
           ["Blend into frame", "CPU", "27.8"],
           ["JPEG encode", "CPU", "23.7"],
           ["Sum of stages in isolation", "", "121.6"],
           ["A real frame, CPU overlapped", "", MEASURED["render_ms"]]],
          widths=[0.5, 0.2, 0.3], size=11.5,
          colors={(2, 2): BAD, (6, 2): GOOD})
    kpi(s, Inches(7.4), Inches(1.9), Inches(2.5), MEASURED["render_speedup"],
        "From overlapping CPU with GPU", f"{MEASURED['render_ms_seq']} → "
        f"{MEASURED['render_ms']} ms/frame", GOOD)
    kpi(s, Inches(10.12), Inches(1.9), Inches(2.5), "74%",
        "Of a frame is VAE decode", "No CPU work left to hide behind it", WARN)
    callout(s, Inches(7.4), Inches(3.55), Inches(5.22), Inches(1.6),
            "Batch size 4 was our own published default, derived on Apple MPS. On CUDA it is "
            "backwards — 16 wins, and the curve is flat past it. float16 is 9.15× float32 with a "
            "mean absolute difference of 0.04 of 255. Both figures were wrong in our docs until "
            "re-measured on the right device.", tone=ACCENT, label="Two figures we retracted")
    callout(s, Inches(0.72), Inches(5.6), Inches(11.9), Inches(1.2),
            "The stage table deliberately no longer sums to the frame cost. GPU stages total "
            f"{MEASURED['gpu_ms']} ms and CPU work totals {MEASURED['cpu_ms']} ms; a real frame "
            f"costs {MEASURED['render_ms']} ms. The gap is the CPU half hiding behind the GPU "
            "half — all but ~8 ms of it recovered, which is the difference between hiding work "
            "and relocating it.", tone=GOOD, label="Reading the table")

    s = content(prs, "What a candidate actually sees", "Metrics · live over WebSocket, 6 turns/run",
                notes="This is a different question from throughput and it needed its own probe. "
                      "Timestamps are taken in the probe process, so they include the socket but "
                      "not a browser's decode or compositor — every figure is a lower bound on "
                      "what a person perceives, which is the safe direction. Say plainly that the "
                      "two runs disagreed in sign on the gap and we report both.")
    table(s, Inches(0.72), Inches(1.9), Inches(11.9),
          ["Metric", "Before", "After", "Note"],
          [["Frames delivered per second", MEASURED["fps_before"],
            MEASURED["fps_delivered_range"], "Against a configured target of 8"],
           ["Frames delivered vs. needed, per turn", "9 of 75", "38 of 45",
            "The shortfall is the start lag, not throughput"],
           ["Trailing audio→video gap, median", "~3 s (recorded, wrong)",
            MEASURED["gap_median"], "Two runs; medians disagree in sign, both reported"],
           ["Worst single turn", "not measured", MEASURED["gap_worst"],
            "Past the ~100 ms a viewer notices"],
           ["Video starts, after audio", "not measured", MEASURED["start_lag"],
            "One render window plus the mixer's lead-in; split unmeasured"],
           ["Frames discarded per turn", MEASURED["discards_before"], MEASURED["discards"],
            "Discarding was always correct — the backlog was not"],
           ["First turn vs. fifth turn", "16.7 s vs 1.4 s", "1.5 s vs 1.5 s",
            "Model cache shared, first forward pass paid at start-up"]],
          widths=[0.28, 0.16, 0.16, 0.4], size=11,
          colors={(0, 2): GOOD, (2, 2): GOOD, (5, 2): GOOD, (6, 2): GOOD})
    callout(s, Inches(0.72), Inches(5.65), Inches(11.9), Inches(1.15),
            "“Trailing gap” means how long video kept arriving after the last audio of the turn. "
            "Negative is healthy — video finished first. Broadcast lip-sync tolerance is about "
            "100 ms, so our median sits at the edge of perceptible and the worst turn is clearly "
            "past it. Six turns per run is not enough to call a mean.",
            tone=ACCENT, label="Definition, and its limits")

    s = content(prs, "The latency budget for a whole turn", "Metrics · full pipeline",
                notes="The conclusion this table exists to support: none of the three dominant "
                      "terms is the renderer. Subtract the renderer entirely and 2.6–5.7 s "
                      "remains. 'We need more GPU' is measurably the wrong diagnosis. The LLM "
                      "figures are a free-tier endpoint — that term is commercial, not "
                      "architectural.")
    table(s, Inches(0.72), Inches(1.9), Inches(11.9),
          ["Stage", "Target", "Measured", "What moves it"],
          [["End-of-turn detection", "100–300 ms", f"{MEASURED['turn_detect']} ms (configured)",
            "Nothing in hardware. Speculative execution trades wasted compute for latency"],
           ["Speech-to-text finalise", "50–150 ms", "~0 observed",
            "Streaming socket — hidden inside the silence window, not added to it"],
           ["LLM time-to-first-token", "200–500 ms", f"{MEASURED['llm_ttft']} ms",
            "Commercial, not architectural: a paid low-latency endpoint"],
           ["TTS time-to-first-audio", "100–300 ms", f"{MEASURED['tts_rest']} ms (REST)",
            f"Aura's WebSocket measured {MEASURED['tts_ws']} ms — verified, largest cheap win"],
           ["Avatar first frame (T4, real face)", "50–150 ms",
            f"{MEASURED['start_lag']} after audio", "Render window plus mixer lead-in"],
           ["Encode + network", "50–150 ms", "20–25 ms",
            "Loopback, so a floor. A real network adds the client's jitter buffer"],
           ["Perceived total", "< 1,000 ms", MEASURED["turn_total"],
            "3–6× target, and the renderer is not why"]],
          widths=[0.25, 0.12, 0.2, 0.43], size=11,
          colors={(2, 2): BAD, (6, 2): BAD})
    callout(s, Inches(0.72), Inches(5.65), Inches(11.9), Inches(1.15),
            "Set the renderer to zero and roughly 2.6–5.7 s of a 2.7–5.8 s turn remains. The part "
            "a vendor sells is the part that was never the problem; the part that is the problem — "
            "turn-taking, cancellation, history truncation, pipelining — is code no vendor API "
            "writes for you. That measurement is the backbone of the build-vs-buy case.",
            tone=GOOD, label="The finding that matters commercially")

    # ---------------------------------------------------------------- 7. rigour
    s = content(prs, "Three bugs, and why they are on a slide", "Engineering rigour",
                notes="Put this in as a credibility slide. All three were the same mistake in "
                      "different places: an assumption true of the stub renderer and false of the "
                      "real one. That is the standing cost of a clean boundary — the stub "
                      "satisfies the Protocol perfectly, so nothing fails until a GPU is behind "
                      "it. Naming it as a class of bug is more useful than three anecdotes.")
    table(s, Inches(0.72), Inches(1.9), Inches(11.9),
          ["Fault", "Symptom", "Root cause"],
          [["Weights reloaded every session",
            "Audio at 6.2 s, first frame at 22.9 s",
            "load() filled an instance attribute while its docstring said “once per process”; "
            "build() returns a fresh backend per session"],
           ["First forward pass unwarmed",
            f"{MEASURED['first_render_ms']} ms for five frames vs {MEASURED['render_ms']} ms "
            "steady",
            "cuDNN algorithm selection, first landmark inference, lazy allocator arenas — none "
            "triggered by loading weights"],
           ["Render ran on the event loop",
            "1.4 fps delivered from a card capable of 12.8",
            "_pump_frames was synchronous on a contract that frames() cannot block — true of the "
            "stub, false of a GPU renderer"]],
          widths=[0.2, 0.28, 0.52], size=11, row_h=0.72)
    callout(s, Inches(0.72), Inches(4.75), Inches(11.9), Inches(1.05),
            "All three were one mistake in three places: an assumption that held for the stub and "
            "not for the real renderer. The stub satisfies the Protocol perfectly, so nothing "
            "about it fails until a GPU is behind it. We now treat that as a class of bug rather "
            "than three incidents.", tone=ACCENT, label="The pattern")
    callout(s, Inches(0.72), Inches(6.0), Inches(11.9), Inches(0.8),
            "The recorded diagnosis for a year was “video lags audio by ~3 s”. Measured properly, "
            "the trailing gap was already near zero. We had been optimising the wrong quantity.",
            tone=GOOD, label="And the premise was wrong")

    # ---------------------------------------------------------------- 8. stack
    section(prs, "05", "Tech stack and models",
            "Everything that runs, what it costs on disk, and what licence it carries.")

    s = content(prs, "Models, by area", "Tech stack",
                notes="Five models cooperate for the face. The landmark model is our own "
                      "substitution: upstream builds an mmpose RTMPose model at import needing "
                      "pinned Linux-only compiled wheels, and it does one line of real work — "
                      "keypoints[0][23:91], the iBUG-68 layout — which face_alignment has "
                      "produced in pure torch since 2017.")
    table(s, Inches(0.72), Inches(1.9), Inches(11.9),
          ["Area", "Model", "Size", "Licence / host", "Role"],
          [["Face · lip-sync", "MuseTalk v1.5 U-Net", "3.24 GB", "MIT code, commercial weights",
            "Repaints the mouth in latent space"],
           ["Face · image codec", "stabilityai/sd-vae-ft-mse", "319 MB", "MIT",
            "256×256 crop ↔ 32×32×4 latents"],
           ["Face · audio encoder", "openai/whisper-tiny (encoder)", "144 MB", "MIT",
            "Conditions the U-Net via cross-attention"],
           ["Face · blending", "BiSeNet + ResNet-18", "95 MB", "MIT",
            "Feathers the generated mouth in"],
           ["Face · landmarks", "face_alignment (FAN + S3FD)", "~180 MB", "BSD-3",
            "Substituted for mmpose RTMPose — see notes"],
           ["Enrollment · animation", "LivePortrait", "2.0 GB", "MIT",
            "Motion transfer onto a still photograph"],
           ["Voice · cloning", "Chatterbox Turbo, 350M", "—", "MIT, self-hosted sidecar",
            "A persona that sounds like a specific person"],
           ["Voice · TTS", "Deepgram Aura 2 (aura-2-thalia-en)", "—", "hosted",
            f"{MEASURED['tts_hosted_ms']} ms to first audio"],
           ["Voice · STT", "Deepgram nova-3", "—", "hosted",
            "Streaming, with endpointing and utterance-end"],
           ["Language", "gpt-oss:20b", "—", "OpenAI-compatible endpoint",
            "Self-hostable without a code change"]],
          widths=[0.16, 0.24, 0.09, 0.22, 0.29], size=10.5, row_h=0.42)

    s = content(prs, "Software, and why there are three environments", "Tech stack",
                notes="The three-venv split is the slide people question, so lead with the "
                      "evidence: installing Chatterbox beside MuseTalk downgraded numpy and torch "
                      "and left libtorch_cuda.so with an undefined ncclCommResume symbol — it "
                      "took the renderer down. No transformers version satisfies both. The "
                      "collision forced a sidecar, which turned out to be the better "
                      "architecture: a second GPU is now configuration.")
    table(s, Inches(0.72), Inches(1.9), Inches(5.85),
          ["Layer", "Stack"],
          [["Runtime", "Python 3.12, FastAPI, uvicorn, pydantic"],
           ["Store", "psycopg 3, PostgreSQL; JSON files as the no-credential default"],
           ["Retrieval", "ChromaDB (optional)"],
           ["ML", "torch 2.13+cu130, diffusers 0.30.2, transformers 4.39.2"],
           ["Vision / audio", "numpy 2.4.6, OpenCV 4.14, librosa, soundfile, ffmpeg 6.1.1"],
           ["Console", "Next.js (React 19), TypeScript, Tailwind, pnpm"],
           ["Assistant", "LangGraph, langchain-core, sse-starlette"],
           ["Media plane", "LiveKit SFU + egress + Redis, via Docker"],
           ["Quality", f"pytest ({MEASURED['tests']}), ruff, mypy strict, Playwright, ESLint"]],
          widths=[0.28, 0.72], size=11)
    table(s, Inches(6.95), Inches(1.9), Inches(5.67),
          ["Environment", "Why it must be separate"],
          [[".venv-musetalk", "numpy 2.4, transformers 4.39.2"],
           [".venv-voice", "Chatterbox needs a newer transformers for LlamaModel"],
           ["LivePortrait/.venv", "pins numpy 1.26 and OpenCV 4.10"]],
          widths=[0.34, 0.66], size=11)
    callout(s, Inches(6.95), Inches(3.5), Inches(5.67), Inches(2.0),
            "This was learned the hard way. Installing Chatterbox beside MuseTalk downgraded numpy "
            "and torch and left libtorch_cuda.so with an undefined ncclCommResume symbol — it took "
            "the renderer down, not the voice. No transformers version satisfies both. Recovery "
            "needed a full uninstall of torch and every nvidia-* package.\n\nThe collision forced "
            "the sidecar, and the sidecar is the better architecture: a hosted API and a local "
            "process look identical to the orchestrator, so a second GPU is configuration rather "
            "than work.", tone=ACCENT, label="Not tidiness — a real collision")

    # ---------------------------------------------------------------- 9. residency
    s = content(prs, "Data residency — where candidate data actually goes", "Compliance posture",
                notes="This is the slide that matters most to the original premise: the vendor "
                      "concern was that candidate audio and video leave your infrastructure. Be "
                      "precise. Today the face never leaves and the conversation does. That is a "
                      "real improvement on a full vendor, and it is not yet the whole promise.")
    table(s, Inches(0.72), Inches(1.9), Inches(11.9),
          ["Component", "Runs", "Candidate data that leaves"],
          [["Renderer (MuseTalk)", "your GPU", "nothing"],
           ["Enrollment (LivePortrait)", "your GPU / CPU", "nothing"],
           ["Voice cloning (Chatterbox)", "your GPU", "nothing"],
           ["Store (Postgres or JSON)", "your host", "nothing"],
           ["Console, assistant, SFU, egress, Redis", "your host / Docker", "nothing"],
           ["Speech to text", "Deepgram", "candidate audio"],
           ["Text to speech", "Deepgram", "interviewer text only"],
           ["Language model", "OpenAI-compatible endpoint", "transcript and prompt"]],
          widths=[0.34, 0.24, 0.42], size=11.5,
          colors={(5, 2): BAD, (7, 2): BAD, (0, 2): GOOD, (1, 2): GOOD, (2, 2): GOOD,
                  (3, 2): GOOD, (4, 2): GOOD})
    callout(s, Inches(0.72), Inches(5.3), Inches(5.8), Inches(1.5),
            "Today the face never leaves and the conversation does. The LLM adapter speaks the "
            "OpenAI wire format precisely so Ollama, vLLM or LM Studio take over with a base-URL "
            "change and no code change.", tone=GOOD, label="What is already true")
    callout(s, Inches(6.82), Inches(5.3), Inches(5.8), Inches(1.5),
            "STT and TTS need real replacements, not configuration — Whisper for one, XTTS-v2 or "
            "F5-TTS for the other. That is genuine work and it is unbuilt. Biometric data is also "
            "now in the store with no authentication in front of it.",
            tone=ACCENT, label="What is not")

    # ---------------------------------------------------------------- 10. production
    section(prs, "06", "Running this in production",
            "What 25 fps costs, what a session occupies, and which of these numbers we have not "
            "earned the right to state.")

    s = content(prs, "What true real time would take", "Production",
                notes="Be concrete about which levers are measured and which are not. The CPU "
                      "overlap is done and returned 1.59×. Everything else on this list is "
                      "unmeasured on our hardware, and the deck says so rather than projecting.")
    table(s, Inches(0.72), Inches(1.9), Inches(11.9),
          ["Lever", "Expected effect", "Evidence"],
          [["Overlap CPU with GPU", f"{MEASURED['render_ms_seq']} → {MEASURED['render_ms']} "
            "ms/frame", "DONE — measured 1.59×"],
           ["A faster VAE decoder (TAESD-class, TensorRT, fp8)",
            "Attacks 74% of the remaining frame", "NOT YET MEASURED — the single largest lever"],
           ["A larger GPU (L4, A10, L40S)", "Shrinks the term that now dominates",
            "NOT YET MEASURED — no card above a T4 tested"],
           ["Lower output resolution", "Does not touch VAE decode — fixed 256×256 work",
            "Measured: proposed and discarded for this reason"],
           ["Bigger batch", "No further gain; the curve is flat past 16",
            "Measured on the T4"],
           ["Second GPU for the voice sidecar", f"Removes {MEASURED['voice_contention']} "
            "contention", "Measured contention; the fix is configuration"]],
          widths=[0.32, 0.34, 0.34], size=11,
          colors={(0, 2): GOOD, (1, 2): WARN, (2, 2): WARN})
    callout(s, Inches(0.72), Inches(5.3), Inches(11.9), Inches(1.5),
            "One T4 serves a self-hosted face or a self-hosted voice, not both: avatar_first_frame "
            f"goes {MEASURED['voice_contention']} when the voice sidecar competes. It is not "
            "memory — 7.6 of 15 GB — it is compute contention, and lowering fps and batch size was "
            "tested and did not help (27,887 ms at 6 fps vs 28,108 ms at 8). So this is a hardware "
            "or vendor decision, not a tuning exercise.",
            tone=ACCENT, label="[HUMAN] the open decision")

    s = content(prs, "Cost per interview — what we can and cannot claim", "Production",
                notes="This is the most important honesty slide in the deck. We have measured "
                      "throughput. We have NOT measured concurrent sessions per GPU, and cost per "
                      "minute is dominated by exactly that. Present the arithmetic as arithmetic, "
                      "name the assumption, and say what one experiment would close it. Do not "
                      "let anyone leave the room quoting a dollar figure as measured.")
    table(s, Inches(0.72), Inches(1.9), Inches(11.9),
          ["Input", "Status", "Value"],
          [["Render cost per frame, T4", "MEASURED", f"{MEASURED['render_ms']} ms"],
           ["Delivered fps, one session, T4", "MEASURED", f"{MEASURED['fps_delivered']} fps"],
           ["GPU memory, one session + models", "MEASURED", "7.6 of 15 GB"],
           ["Concurrent sessions per T4", "NOT YET MEASURED",
            "Never tested above one. This is the term cost per minute depends on most"],
           ["Enrollment cost, one persona", "MEASURED",
            f"~{MEASURED['prepare_ms']} ms GPU, once per persona, then cached"],
           ["Cloud GPU list price", "ASSUMPTION — not a quote",
            "PROCESS.md §4.2 assumes a T4-class instance at ~$0.35–0.50/hr"],
           ["Vendor per-minute price", "ASSUMPTION — no contract in hand",
            "PROCESS.md §4.2 assumes ~$0.10–0.30/min at list"]],
          widths=[0.3, 0.22, 0.48], size=11,
          colors={(0, 1): GOOD, (1, 1): GOOD, (2, 1): GOOD, (4, 1): GOOD,
                  (3, 1): BAD, (5, 1): WARN, (6, 1): WARN})
    callout(s, Inches(0.72), Inches(5.4), Inches(11.9), Inches(1.4),
            "We deliberately do not put a $/minute figure on a slide. The arithmetic in "
            "PROCESS.md §4.2 lands one to two orders of magnitude below the assumed vendor price — "
            "but it divides a measured GPU cost by an UNMEASURED concurrency number, and that "
            "number is the whole answer. One experiment closes it: run N concurrent sessions on "
            "one card and find where fps falls below the target. Until then the direction is "
            "credible and the magnitude is not.",
            tone=ACCENT, label="Why there is no dollar figure on this slide")

    s = content(prs, "Observability — where we instrument", "Production",
                notes="Telemetry is not a side channel here: it is the only recorder of a turn. "
                      "The server builds the stored transcript from the event stream rather than "
                      "from separate write calls, so there is exactly one authority and a second "
                      "path cannot disagree with it. The cost of that design is that an unemitted "
                      "event is an unrecorded turn — which is precisely the silence-re-prompt bug.")
    table(s, Inches(0.72), Inches(1.9), Inches(11.9),
          ["Signal", "Emitted at", "Answers"],
          [["latency: turn_detect", "End-of-turn decision", "Is the silence window right for "
            "this candidate?"],
           ["latency: llm_ttft", "First sentence from the model", "Is the endpoint or the prompt "
            "the problem?"],
           ["latency: tts_first_audio", "First PCM chunk", "REST vs WebSocket, and voice choice"],
           ["latency: avatar_first_frame", "Browser paint, not socket write",
            "The only end-to-end number a candidate feels"],
           ["latency: perceived_total", "Turn start to first paint", "The headline SLO"],
           ["frames_repeated / frames_discarded", "Mixer", "Is the renderer keeping cadence, or "
            "is the queue draining?"],
           ["stale_artifact_dropped", "Epoch check", "Is cancellation actually working?"],
           ["heard (with transcribed / silent)", "Turn open", "Broken STT vs. a quiet candidate — "
            "different pages"],
           ["WarmupReport, schema_problems", "/config", "Is this process ready, and is the "
            "database current?"]],
          widths=[0.28, 0.24, 0.48], size=11)

    # ---------------------------------------------------------------- 11. judgment
    section(prs, "07", "Recommendation and migration",
            "These sections belong to the author. What follows is quoted from PROCESS.md, with "
            "the one input that changed since it was written.")

    s = content(prs, "Build vs. buy — the standing recommendation", "Quoted from PROCESS.md §4",
                notes="Read the recommendation as written; do not extend it. Then be explicit "
                      "that one of its inputs changed: §4.2 said 'I did not run a GPU', and now "
                      "one has run. The measured render cost is far below the assumed vendor "
                      "price, which if anything strengthens the case — but re-deriving the memo "
                      "is the author's judgment call, not something to assert from a slide.")
    callout(s, Inches(0.72), Inches(1.9), Inches(11.9), Inches(1.5),
            "“Keep the vendor for the rendering stage. Build the orchestration layer in-house, "
            "starting now.” That is the hybrid, and the split is not a hedge — it falls directly "
            "out of what was measured.", tone=GOOD, label="The recommendation, as written")
    bullets(s, Inches(0.72), Inches(3.6), Inches(11.9), [
        ("The reasoning, unchanged: the model was never the bottleneck.",
         f"A full turn measures {MEASURED['turn_total']} against a sub-second target, and the "
         "three dominant terms are the end-of-turn policy, LLM time-to-first-token and TTS. The "
         "part a vendor sells is the part that was never the problem."),
        ("What changed since it was written: a GPU ran.",
         "§4.2 opens “I have no vendor contract and did not run a GPU.” Render cost per frame is "
         "now measured, and it lands well below the assumed vendor per-minute price — but it "
         "divides by a concurrency figure we still have not measured."),
        ("[HUMAN] The refresh this deserves, and who owns it.",
         "Re-derive §4.2 with the measured render cost, run the concurrency experiment, and "
         "restate §4.4's thresholds. The recommendation may survive intact; asserting that from a "
         "slide would be exactly the foregone conclusion the brief warns against."),
    ], size=13)

    s = content(prs, "Migration plan — cutover without a regression", "Quoted from PROCESS.md §5",
                notes="The test of this section in the brief is whether another senior engineer "
                      "could execute it without the author in the room. Walk the four phases and "
                      "point at the preconditions — the plan is gated on things being true before "
                      "phase 1 starts, which is what makes it executable rather than aspirational.")
    flow(s, Inches(0.72), Inches(1.95), Inches(11.9), [
        ("Preconditions", "Named and checkable before anything ships. A cutover gated on nothing "
         "is a hope."),
        ("Phase 1 · Shadow", "Render in parallel, serve the vendor. Compare quality and latency "
         "on real traffic, no candidate affected."),
        ("Phase 2 · Flagged rollout", "Per-tenant flag, small cohorts first, with the abort "
         "criteria written down in advance."),
        ("Rollback", "One flag flip back to the vendor. The vendor path stays warm and paid for "
         "until decommission."),
        ("Decommission", "Only after a defined quiet period with no regression. Removing the "
         "fallback is the last step, not the first."),
    ])
    bullets(s, Inches(0.72), Inches(3.5), Inches(11.9), [
        ("The boundary is what makes this cheap.",
         "The renderer is a Protocol with two implementations already. Shadow mode is a second "
         "implementation of an interface that exists, not a fork of the pipeline — which is the "
         "practical payoff of the contracts decision on slide 9."),
        ("Rollback is a flag, not a redeploy.",
         "Because the vendor and in-house renderers are interchangeable at the same seam, and "
         "because session state lives in the orchestrator rather than in the renderer."),
    ], size=13)

    # ---------------------------------------------------------------- 12. gaps
    s = content(prs, "What is missing — stated, not buried", "Risks and gaps",
                notes="Close the deck on this rather than on the recommendation. Volunteering the "
                      "gap list is the argument for the rest of the deck being trustworthy. The "
                      "authentication row is the one to say slowly: the store now holds a real "
                      "person's face and voice, which changes the posture from untidy to serious.")
    table(s, Inches(0.72), Inches(1.9), Inches(11.9),
          ["Gap", "Consequence", "Standing"],
          [["No authentication anywhere", "The candidate link is not a credential; the assistant "
            "reads any transcript. The store now holds real faces and voices — biometric data",
            "Deferred by the owner"],
           ["One GPU cannot host face + voice", "Self-hosted cloning and a self-hosted face are "
            "an either/or ({MEASURED['voice_contention']})", "[HUMAN] decision"],
           ["Concurrency per GPU untested", "Cost per minute cannot be credibly stated",
            "One experiment away"],
           ["2.0× short of 25 fps", f"{MEASURED['render_ms']} ms/frame; VAE decode is 74%",
            "Needs a faster decoder or a bigger card"],
           ["Start lag split unmeasured", f"{MEASURED['start_lag']} before video starts — render "
            "window vs mixer lead-in unknown", "Measure before tuning"],
           ["Self-hosted STT/TTS unbuilt", "The conversation still leaves your infrastructure",
            "Real work, not configuration"],
           ["No output-quality comparison", "Our substituted landmark detector vs mmpose RTMPose "
            "is unmeasured for quality", "Known, unquantified"],
           ["No CUDA float32 ratio", "The fp16 default is not in doubt; the ratio on CUDA is",
            "Low value, stated anyway"]],
          widths=[0.24, 0.48, 0.28], size=10.5, row_h=0.44,
          colors={(0, 2): BAD, (1, 2): WARN, (2, 2): WARN})

    s = content(prs, "How to verify any of this yourself", "Reproducibility",
                notes="End by handing over the commands. Everything on the metrics slides came "
                      "from one of these, and each writes a JSON file alongside its console "
                      "output so a figure can be traced rather than trusted.")
    table(s, Inches(0.72), Inches(1.9), Inches(11.9),
          ["Command", "Produces"],
          [["pytest -m \"not gpu\"", f"{MEASURED['tests']} tests: no GPU, no weights, no network"],
           ["python scripts/bench_renderer.py", "The throughput and stage tables, plus the "
            "sequential-vs-overlapped comparison"],
           ["python scripts/measure_lag.py --agent <id>", "The live delivery figures: fps, "
            "trailing gap, start lag, discards"],
           ["python scripts/smoke_session.py", "17 assertions over a real socket: barge-in, "
            "epoch drops, cadence, mic-driven turns"],
           ["curl localhost:8000/config", "Which implementation each boundary resolved to, warm-up "
            "state, and schema problems"],
           ["docker compose --env-file .env.development up -d", "SFU, egress and Redis — recording "
            "needs a recorder the SFU binary omits"]],
          widths=[0.42, 0.58], size=11.5, row_h=0.46)
    callout(s, Inches(0.72), Inches(5.4), Inches(11.9), Inches(1.2),
            "MEASUREMENTS.md is the source of every number in this deck, including a §9 that lists "
            "what is not measured and a record of two figures we published and later retracted "
            "because they had been measured on the wrong device. The retraction is in the document "
            "on purpose.", tone=GOOD, label="Where the numbers live")

    title_slide(prs, "Ends", "Clarity over cleverness.\nBoundaries over convenience.\nProof over opinion.",
                "Every figure in this deck came from a run on named hardware. Where a number does "
                "not exist, the slide says NOT YET MEASURED and says what would close it.")

    prs.save(str(out))
    print(f"wrote {out} — {len(prs.slides.__iter__.__self__._sldIdLst)} slides")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="nod-engineering-review.pptx")
    args = parser.parse_args()
    build(Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
