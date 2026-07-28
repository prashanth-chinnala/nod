# nod

A prototype real-time conversational avatar: audio in, lip-synced talking-head video
out, streamed to a browser, with session lifecycle and interruption handling.

Built as the Exterview Head of Engineering take-home. The reasoning, the
model-selection memo, the build-vs-buy recommendation, and the migration plan live in
[PROCESS.md](PROCESS.md); this file is only how to run what exists.

> **Status: M1 and M3 complete, M0 blocked.** The session layer and the streaming path
> work end to end. There is **no ML model wired in**, so there is no video of a face —
> the avatar is a coloured rectangle that changes in time with the audio. See
> [What works today](#what-works-today) for the exact boundary and
> [DEVLOG.md](DEVLOG.md) for what was deferred and why.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,server]"
uvicorn avatar.server:app
```

Then open **<http://127.0.0.1:8000>** and press **Start session**.

- **Starts speaking** → the session moves to `LISTENING`
- **Stops speaking** → `THINKING`, then `SPEAKING` with audio and frames
- **Starts speaking** again mid-answer → barge-in. Watch the epoch increment, the audio
  cut off, the log fill with stale-frame drops, and the state return to `LISTENING`

Every number on that page is measured. The one at `/mockup` is the design reference with
invented values, kept for comparison.

To verify the same thing headlessly:

```bash
python scripts/smoke_session.py     # 15 assertions over a real socket
```

## What works today

| | |
|---|---|
| Session state machine — start/stop, listening, thinking, speaking, cancelling | Working, tested |
| Barge-in via turn-epoch invalidation, stale artifacts provably dropped | Working, verified end-to-end |
| Constant-cadence frame mixer, idle-loop fallback, starvation handling | Working, tested. Measured at 25.4fps |
| WebSocket streaming of frames and audio to a browser | Working |
| Browser client — canvas, Web Audio, mic capture, live telemetry | Working, no build step |
| History truncated to audio the client **acknowledged playing** | Working, tested |
| End-to-end latency measured to browser paint, not to socket write | Working |
| A renderer behind a Protocol, with a GPU-free implementation | Working, tested |
| **A talking-head model of any kind** | **Not built — blocked on M0, needs a GPU** |
| **Real STT, LLM, TTS** | **Not built (M4).** Placeholders with real timing, fake content |
| **Real turn detection (VAD)** | **Not built (M4).** A client-side energy gate stands in |
| **Frame encoding (JPEG/WebP)** | **Not built (M2).** Uncompressed BMP, ~2.7MB/s |

The headline numbers the brief asks for — first-frame latency and fps for a real
talking-head model — read `NOT YET MEASURED` in [PROCESS.md](PROCESS.md) §3.3, because
they do not exist yet. §3.3.1 has the session-layer numbers that do, each with a note on
what it actually measures.

## Requirements

Python 3.11 or newer. **No GPU, no model weights, and no network** are needed for
anything currently in the repo.

A GPU becomes a requirement at M2, when the real renderer lands. Target hardware is a
Colab/Kaggle free-tier T4; the weight-download step and real hardware requirements get
documented here once M0 has actually run.

## Run the checks

```bash
pytest -m "not gpu"                        # 131 tests, ~0.1s, no GPU
ruff check src tests && ruff format --check src tests
mypy src/avatar
```

That is exactly what [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs, and it
installs only `[dev]` — pytest, ruff, mypy, no ML dependency and no web stack. The suite
is fast because every collaborator is injected and the clock is faked, so a test that
needs a simulated second gets it in microseconds.

## Layout

```
src/avatar/
  contracts.py         dataclasses + the four Protocols. Imports nothing from the package.
  state.py             State enum, transition table, frame-source table. All data.
  orchestrator.py      SessionOrchestrator — every state transition lives here
  mixer.py             FrameMixer, IdleLoop — cadence and presentation timestamps
  telemetry.py         emit call sites; structured JSON, subscribable
  idle.py              placeholder idle loop + loader for a real prepared clip
  llm.py               sentence chunker + scripted interviewer
  bmp.py               twenty-line BMP encoder, so nothing needs Pillow
  server.py            FastAPI. The only module that imports a web framework.
  audio/tts.py         ToneTTS — real timing, fake voice
  transport/websocket.py   wire codec + Transport. No framework dependency.
  renderers/           build() registry + StubRenderer (no GPU, no deps)
tests/                 131 tests, including the boundary enforcement
scripts/               headless end-to-end verification
web/index.html         the real client, measured numbers
web/mockup.html        design mockup, simulated numbers
```

## The module boundary

Nothing in `contracts.py`, `state.py`, `telemetry.py`, `mixer.py`, or
`orchestrator.py` may import torch, CUDA, or a renderer implementation. That is what
keeps CI GPU-free, and it is the mechanical proof that the ML model is one bounded,
swappable piece rather than a claim to that effect.

[`tests/test_boundaries.py`](tests/test_boundaries.py) enforces it by parsing each
module's imports from its AST, so the assertion is that the dependency is absent from
the graph — not merely that it happened not to be installed. Swapping the model is a
change to one `RendererConfig` value, and that too is a test.

If CI ever needs a GPU package in order to import, a boundary has been broken. Fix the
boundary; do not add the dependency.
