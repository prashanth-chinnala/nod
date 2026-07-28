# CLAUDE.md

## Before starting any task

Read `IMPLEMENTATION_GUIDE.md`. It contains the milestone plan, the decisions already
made, the test specifications, and the list of things you must not do.

## Standing rules

1. **Never invent a measurement.** No latency, fps, VRAM, or cost figure may be
   written unless it came from an actual run. If a number is needed and no run
   exists, write `NOT YET MEASURED`. This project is graded on honesty about
   constraints; a plausible fabricated number is the worst possible outcome.

2. **Judgment sections belong to the human.** Do not write the build-vs-buy
   recommendation, the confirmed-vs-inferred tags, or the "what would change my mind"
   thresholds. Scaffold them, mark `[HUMAN]`, report the gap.

3. **Respect the module boundary.** Nothing in `src/avatar/orchestrator.py`,
   `mixer.py`, or `state.py` may import torch, CUDA, or any renderer implementation.
   `contracts.py` imports nothing from the package. If you need to break this, stop
   and ask — the boundary is a graded requirement, not a style preference.

4. **Finish the current milestone before starting the next.** Acceptance criteria are
   in the guide. A half-finished everything is the main failure mode here.

5. **Boring over clever.** A well-chosen existing library beats a novel approach.
   Prefer the option that will still work on submission day.

## Working style

- Commit at every coherent step with a message explaining *why*. Never squash.
- Append to `DEVLOG.md` at the end of each session: attempted, worked, deferred + why.
- Run `ruff check src tests && mypy src/avatar && pytest -m "not gpu"` before
  declaring any task done.
- Do not commit model weights, media outputs, virtualenvs, or secrets.

## Commands

```bash
pip install -e ".[dev]"              # setup
pytest -m "not gpu"                  # CI-equivalent test run
pytest                               # includes GPU-marked tests
ruff check src tests && mypy src/avatar
uvicorn avatar.server:app --reload   # run the prototype (M3+)
python scripts/measure_latency.py    # produces numbers for PROCESS.md
```

## Current state

> Update this section at the end of each session.

- Active milestone: **M1 — skeleton, contracts, stub renderer, CI green**
- Model chosen: *not yet decided; M0 blocked on hardware. See `DEVLOG.md`.*
- Blocked on: *M0 model spike needs a GPU; M2 cannot start until M0 resolves.*
