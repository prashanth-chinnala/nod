# nod

A prototype real-time conversational avatar: audio in, lip-synced talking-head video
out, streamed to a browser, with session lifecycle and interruption handling.

Built as the Exterview Head of Engineering take-home. The reasoning, the
model-selection memo, the build-vs-buy recommendation, and the migration plan live in
[PROCESS.md](PROCESS.md); this file is only how to run what exists.

> **Status: the session layer is complete; the talking-head model is not integrated.**
> With credentials this runs a real spoken conversation today — real transcription, a real
> LLM, a real synthesised voice, and working interruption. What is missing is the **face**:
> no talking-head model is wired in, so the avatar is five rectangles whose mouth height
> tracks the audio in real time. See [What works today](#what-works-today) for the exact
> boundary and [PROCESS.md](PROCESS.md) §3.3.3 for the measured latency budget.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,server]"
uvicorn avatar.server:app
```

Then open **<http://127.0.0.1:8000>** and press **Start session**.

That works with **no credentials at all** — every default is a placeholder that needs no
key and no network. To hear a real voice and hold a real conversation, see
[Configuration](#configuration).

- **Starts speaking** → the session moves to `LISTENING`
- **Stops speaking** → `THINKING`, then `SPEAKING` with audio and frames
- **Starts speaking** again mid-answer → barge-in. Watch the epoch increment, the audio
  cut off, the log fill with stale-frame drops, and the state return to `LISTENING`

Every number on that page is measured. No simulated variant of the page is kept in the
repo, so there is nothing that could be mistaken for it.

To verify the same thing headlessly:

```bash
python scripts/smoke_session.py     # 17 assertions over a real socket
```

That script drives a real session over a real socket and asserts the properties a
screenshot cannot show: that presentation timestamps are strictly monotonic, that
stale-epoch artifacts were provably dropped rather than merely overtaken, and that
end-to-end latency was measured to browser paint rather than to the socket write.

## Configuration

Every component is chosen by an environment variable, and **every default is a working
no-credential one** — a clean clone runs with no env file at all, on placeholders for the
LLM, TTS, transcriber, and renderer. Create a file with whichever services you have:

```bash
printf 'AVATAR_LLM=openai\nDEEPGRAM_API_KEY=...\n' > .env.development
chmod 600 .env.development      # every .env* is gitignored; none may be committed
```

It is loaded automatically at server import — see
[`src/avatar/config.py`](src/avatar/config.py). No `source` step, no `python-dotenv`.
Three candidates are read in descending precedence — `.env.development`, `.env.local`,
`.env` — so shared defaults can live in one file and the handful you are changing in
another, without duplicating the rest. `AVATAR_ENV_FILE=/path/to/file` skips the search
entirely, for a mounted secret or a path outside the repo.

**A real environment variable always wins over every file**, so
`AVATAR_TTS=tone uvicorn ...` still overrides, and CI cannot be clobbered by a stray file.

`GET /config` reports which implementation each boundary resolved to, and which variable
*names* came from the file — never their values.

| Variable | Default | Options |
|---|---|---|
| `AVATAR_RENDERER` | `stub` | `stub` (no GPU) · the chosen model, once it is integrated |
| `AVATAR_LLM` | `scripted` | `scripted` · `openai` · `anthropic` |
| `AVATAR_TTS` | `tone` | `tone` · `deepgram` |
| `AVATAR_STT` | `none` | `none` · `deepgram` |
| `AVATAR_VAD` | `energy` | `energy` (no deps) · `silero` (needs `.[vad]`, **never executed**) |
| `AVATAR_LLM_MODEL` | adapter default | any model the chosen endpoint serves |
| `AVATAR_TTS_VOICE` | `aura-2-thalia-en` | any Aura voice |
| `OPENAI_BASE_URL` | vendor default | any OpenAI-compatible endpoint — see below |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `DEEPGRAM_API_KEY` | unset | credentials |

### Any OpenAI-compatible endpoint

`AVATAR_LLM=openai` plus `OPENAI_BASE_URL` reaches far more than OpenAI, because Ollama,
LM Studio, and vLLM all speak the same wire format. **A local model therefore needs no new
adapter and no key at all** — which makes a fully offline, zero-cost interviewer a config
change:

| Target | `OPENAI_BASE_URL` | Key |
|---|---|---|
| Local Ollama | `http://localhost:11434/v1` | none needed |
| Ollama Cloud | `https://ollama.com/v1` | your Ollama key |
| LM Studio | `http://localhost:1234/v1` | none needed |
| OpenAI | *(omit)* | your OpenAI key |

### Running it for real

```bash
# .env.development holds the keys; no prefixes needed
uvicorn avatar.server:app
curl -s localhost:8000/config | python3 -m json.tool   # confirm what resolved
```

If `/config` shows `scripted`, `tone`, or `none` when you expected otherwise, no env file
was picked up and you are measuring placeholders. Its `env_files_read` field names the
files that were actually read, which is the fastest way to tell "the value is wrong" from
"the file was never opened".

### Secrets

**No env file is tracked, and none may become one.** `.gitignore` covers `.env` and every
`.env.*`; nothing is exempted. This repository is published, and a key committed to a
public repo is scraped within minutes and cannot be un-published — rewriting history does
not help, because the crawlers already have it.

There is deliberately no `.env.example`: a committed template is one `git add -f` away from
being a committed key, and the table above already documents every variable. Worth running
yourself rather than taking on trust:

```bash
git ls-files | grep -E '^\.env'      # must print nothing
```

For Colab, use **Colab Secrets** rather than a notebook cell — anything typed into a cell
is saved inside the notebook file.

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
| Server-side turn-taking policy — onset, hysteresis, retraction, end-of-turn | Working, 30 tests |
| Real transcription (Deepgram Nova) | Working. Transcribed 5.48s of real speech exactly |
| Real voice (Deepgram Aura-2) | Working. ~380ms warm time-to-first-audio |
| Real LLM — two adapters, any OpenAI-compatible endpoint | Working via Ollama Cloud |
| `.env` loaded automatically; `GET /config` reports what resolved | Working |
| **A talking-head model of any kind** | **Not built** — needs a GPU; the spike failed in setup (PROCESS.md §2.2.1) |
| **A real voice activity detector** | **Partly.** The policy is real and tested; the detector under it is an energy gate. `SileroVad` is written and **never executed** |
| Frame encoding | Working. PNG, stdlib zlib. **108.10 KB → 0.57 KB per frame, 22.2 → 0.12 Mbps** |
| Client jitter buffer | Working. 150ms lead, underruns counted and surfaced |

The headline numbers the brief asks for — first-frame latency and fps for a real
talking-head model — read `NOT YET MEASURED` in [PROCESS.md](PROCESS.md) §3.3, because
they do not exist yet. §3.3.1–3.3.3 have the numbers that do, each with a note on what it
actually measures.

**The measurement that matters most** (§3.3.3, every component real): a full conversational
turn takes **3.7–5.4s** against a sub-second target — and **none of the three dominant
terms is the renderer.** End-of-turn detection is 700ms of deliberate policy, LLM
time-to-first-token 1.9–3.2s, TTS time-to-first-audio 0.9–1.3s. A perfect zero-latency
talking-head model would still leave ~3.4s, so "more GPU" demonstrably does not close this
gap.

## Requirements

Python 3.11 or newer. **No GPU, no model weights, and no network** are needed for
anything currently in the repo.

A GPU becomes a requirement only when the talking-head model is integrated. The intended
target is a Colab/Kaggle free-tier T4; the weight-download step and the real hardware
requirements will be documented here once the model spike has actually produced throughput
figures. It has not — see [PROCESS.md](PROCESS.md) §2.2.1.

## Run the checks

```bash
pytest -m "not gpu"                        # 199 tests, ~0.6s, no GPU
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
  contracts.py         dataclasses + the five Protocols. Imports nothing from the package.
  state.py             State enum, transition table, frame-source table. All data.
  orchestrator.py      SessionOrchestrator — every state transition lives here
  mixer.py             FrameMixer, IdleLoop — cadence and presentation timestamps
  telemetry.py         emit call sites; structured JSON, subscribable
  idle.py              placeholder idle loop + loader for a real prepared clip
  llm.py               sentence chunker + scripted interviewer
  bmp.py               twenty-line BMP encoder, so nothing needs Pillow
  server.py            FastAPI. The only module that imports a web framework.
  config.py            loads .env at import; a real env var always wins
  llm_anthropic.py     Claude adapter + the LLM registry
  llm_openai.py        OpenAI adapter — also Ollama / LM Studio / vLLM via base_url
  audio/turn_detection.py  onset / hysteresis / retraction / end-of-turn. Pure policy.
  audio/vad.py         EnergyVad (no deps) + SileroVad (torch, never executed)
  audio/tts.py         ToneTTS — real timing, fake voice
  audio/tts_deepgram.py    Aura. container=none matters; see the docstring
  audio/stt.py         Deepgram Nova. Transcribes; decides nothing
  transport/websocket.py   wire codec + Transport. No framework dependency.
  renderers/           build() registry + StubRenderer (no GPU, no deps)
tests/                 225 tests, including the boundary enforcement
scripts/               headless end-to-end verification
web/index.html         the real client, measured numbers
notebooks/             model spike harness, and running the server on a cloud GPU
```

## Documentation map

| File | What it is |
|---|---|
| [PROCESS.md](PROCESS.md) | Architecture document, model-selection memo, build-vs-buy memo, and migration plan |
| [notebooks/m0_musetalk_v2.ipynb](notebooks/m0_musetalk_v2.ipynb) | The model spike harness. Gates on imports, audits every checkpoint, and refuses to report fps without an output file |
| [notebooks/run_on_colab.ipynb](notebooks/run_on_colab.ipynb) | Runs the server on a cloud GPU behind an HTTPS tunnel |

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
