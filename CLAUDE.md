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

- Milestones done: **M1** (contracts, state machine, stub renderer, CI), **M3**
  (WebSocket streaming, browser client), **M4's turn-taking** (server-side VAD, onset /
  hysteresis / retraction / end-of-turn policy), and the real STT, TTS, and LLM adapters.
  **199 tests, all GPU-free. 17/17 end-to-end with every real component live.**
- Repo is public: <https://github.com/prashanth-chinnala/nod>
- Active milestone: **M0 — model spike.** Run 1 failed in setup; see `docs/M0_HOW_TO.md`
  and `notebooks/m0_musetalk_v2.ipynb`.
- Hardware: **Colab / Kaggle free tier (T4 16GB)**, confirmed available. **The pick itself
  is `[HUMAN]`** — see `IMPLEMENTATION_GUIDE.md` §2 and Rule 2.
- Blocked on: *M0 only.* M2 waits on it. Nothing else does.

## Running it

```bash
pip install -e ".[dev,server,tts,llm]"
uvicorn avatar.server:app                 # then open http://127.0.0.1:8000
curl -s localhost:8000/config             # which implementation each boundary resolved to
python scripts/smoke_session.py           # headless end-to-end check, 17 assertions
```

Every default is a working no-credential one, so that runs on a clean clone with no env
file. Credentials go in `.env.development` (or `.env.local` / `.env`, in that descending
precedence), loaded at server import by `src/avatar/config.py` — no `source` step. A real
environment variable always beats every file. Every `.env*` is gitignored with no
exemption; `/config`'s `env_files_read` names which were read, never their contents. `docs/DEMO_SCRIPT.md` is the manual test
protocol and doubles as the Loom shot list.
