# nod

A prototype real-time conversational avatar: audio in, lip-synced talking-head video
out, streamed to a browser, with session lifecycle and interruption handling.

Built as the Exterview Head of Engineering take-home. The reasoning, the
model-selection memo, the build-vs-buy recommendation, and the migration plan live in
[PROCESS.md](PROCESS.md); this file is only how to run what exists.

> **Status: milestone M1 of 7.** The session layer is complete and tested. There is
> no ML model wired in yet, and therefore no video of a face. See
> [What works today](#what-works-today) for the honest boundary, and
> [DEVLOG.md](DEVLOG.md) for what was deferred and why.

## What works today

| | |
|---|---|
| Session state machine — start/stop, listening, thinking, speaking, cancelling | Working, tested |
| Barge-in via turn-epoch invalidation, with stale artifacts provably dropped | Working, tested |
| Constant-cadence frame mixer with idle-loop fallback and starvation handling | Working, tested |
| History truncated to audio the client acknowledged playing | Working, tested |
| Instrumentation call sites for every stage of the latency budget | Working, tested |
| A renderer behind a Protocol, with a GPU-free implementation | Working, tested |
| **An actual talking-head model** | **Not yet — blocked on M0, needs a GPU** |
| **Streaming to a browser** | **Not yet — M3** |
| **STT / LLM / TTS** | **Not yet — M4. Injected as Protocols; no implementations** |

No latency, fps, or VRAM number appears anywhere in this repo, because none has been
measured yet. `PROCESS.md` says `NOT YET MEASURED` where those numbers will go.

## Requirements

- Python 3.11 or newer. No GPU, no model weights, and no network are needed for
  anything currently in the repo.

A GPU becomes a requirement at M2, when the real renderer lands. Hardware
requirements and the weight-download step will be documented here at that point.

## Setup

```bash
git clone <this repo> && cd nod
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

`[dev]` is pytest, ruff, and mypy. It contains no ML dependency, deliberately — see
[The module boundary](#the-module-boundary).

## Run the checks

```bash
pytest -m "not gpu"                        # 89 tests, ~0.1s, no GPU
ruff check src tests && ruff format --check src tests
mypy src/avatar
```

That is exactly what [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs. The
suite is fast because every collaborator is injected and the clock is faked, so a test
that needs a simulated second gets it in microseconds.

## Look at the design mockup

```bash
open web/mockup.html
```

Every number in it is simulated. It fixes the layout and the state vocabulary for the
real client, which lands in M3 as `web/index.html`. The mockup is the specification
for what the demo instrument should surface — first-frame latency, `frames_repeated`,
the per-stage waterfall, and the turn epoch.

## Layout

```
src/avatar/
  contracts.py     dataclasses + the four Protocols. Imports nothing from the package.
  state.py         State enum, transition table, frame-source table. All data.
  orchestrator.py  SessionOrchestrator — every state transition lives here
  mixer.py         FrameMixer, IdleLoop — cadence and presentation timestamps
  telemetry.py     emit call sites; structured JSON in the prototype
  renderers/       build() registry + StubRenderer (no GPU, no deps)
tests/             89 tests, including the boundary enforcement
web/mockup.html    design mockup, simulated data
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
