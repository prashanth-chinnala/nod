# nod

A real-time conversational avatar for technical interviews. A candidate opens a link, sees a face,
and has a conversation: the interviewer listens, decides what to ask next from a competency plan,
speaks in a real voice, and is scored afterwards against a rubric.

Self-hosted. The renderer, the store and the console all run on hardware you control.

**Current state:** working end to end on an NVIDIA T4 with a real face, a real voice and real
transcription. Not yet real time — measured at 12.8 fps against a 25 fps target, and this repository
says so rather than rounding up. Every figure in [MEASUREMENTS.md](MEASUREMENTS.md) came from a run;
where a number does not exist, that file says so.

---

## What it does

| | |
|---|---|
| **Two-way conversation** | streaming transcription, barge-in, turn detection tuned per agent |
| **A real face** | MuseTalk repaints the mouth of a reference clip you upload, in step with the speech |
| **Asks what matters** | a competency plan chooses the next question and tracks coverage |
| **Scores afterwards** | an asynchronous judge produces labelled verdicts with verified quotes |
| **Reviewable** | full transcript, per-stage latency, and an optional H.264 recording |
| **A console** | agents, faces, rubrics, knowledge, guardrails, pronunciations, sessions, reports |
| **An assistant** | asks questions across interviews and proposes rubric changes a human commits |

---

## Documentation

| | |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | the boundaries, the interview loop, why cancellation is an integer |
| [MODELS.md](MODELS.md) | every model, why it was chosen over the alternatives, what would replace it |
| [MEASUREMENTS.md](MEASUREMENTS.md) | every measured figure with its device — and the gaps, named |
| [OPERATIONS.md](OPERATIONS.md) | running it on a laptop and on a GPU host, and what to expect |
| [SECURITY.md](SECURITY.md) | the auth posture, stated rather than implied, and what real faces change |
| [ROADMAP.md](ROADMAP.md) | done, next, and deliberately deferred |
| [PROCESS.md](PROCESS.md) | the engineering log: what was tried, what failed, and why |

---

## Run it

### Without a GPU

Every default is a working, credential-free one, so this runs on a clean clone with no env file.

```bash
pip install -e "apps/api[dev,server,tts,llm]"
uvicorn avatar.server:app                     # API on :8000
curl -s localhost:8000/config                 # which implementation each boundary resolved to
python apps/api/scripts/smoke_session.py      # headless end-to-end check, 17 assertions
```

```bash
cd apps/web && pnpm install && pnpm dev        # console on :3000
```

You get the stub renderer — a placeholder face driven by audio amplitude — a sine-wave voice and a
scripted interviewer. Everything else is real: the state machine, turn taking, barge-in, the
competency plan, the store and the scoring path.

`smoke_session.py` asserts the things a screenshot cannot show — that presentation timestamps are
strictly monotonic, that stale-epoch artifacts were provably dropped rather than merely overtaken,
and that latency was measured to browser paint rather than to the socket write.

### With a GPU, and a real face

```bash
cd apps/api
./scripts/setup_musetalk.sh                    # venv, MuseTalk, 3.7 GB of weights, all verified
AVATAR_RENDERER=musetalk uvicorn avatar.server:app
```

Then upload a reference on the console's **Faces** screen — a clip of one person sitting still and
looking at the camera, 20 seconds or more — press Prepare, and attach it to an agent.

`setup_musetalk.sh` opens with `set -euo pipefail`, and that is the point. It replaces upstream's
`download_weights.sh`, which has no `set -e` and ends by printing "✅ All weights have been
downloaded successfully!" unconditionally — which is how the first attempt finished with 96 MB on
disk and exit code 0. `scripts/fetch_musetalk_weights.py` opens and checks every artifact:
HTML-page detection first, because that names the cause, then a size floor, then a container-format
check. Nothing is reported present unless it was read.

See [OPERATIONS.md](OPERATIONS.md) for a GPU host and the numbers to expect.

### Configuration

Credentials go in `.env.development` (or `.env.local`, or `.env`, in descending precedence), loaded
at server import — there is no `source` step. A real environment variable always beats every file.
Every `.env*` is gitignored with no exemption; `/config` reports which files were read, never their
contents.

| | |
|---|---|
| `AVATAR_RENDERER` | `stub` or `musetalk` |
| `AVATAR_FPS` | target frame rate. 25 is the default and the target; lower it to match the hardware |
| `AVATAR_STORE` | unset for JSON files, `postgres` for Postgres |
| `AVATAR_LLM` / `AVATAR_TTS` / `AVATAR_STT` | which provider each boundary resolves to |
| `AVATAR_MEDIA_DIR` / `AVATAR_DATA_DIR` | **set these absolutely.** Both default to relative paths, and two services started from different directories will disagree about where the data is |

---

## Tests

```bash
cd apps/api && pytest -m "not gpu"             # 746 tests: no GPU, no weights, no network
ruff check src tests && mypy src/avatar
```

That property is enforced rather than hoped for. `tests/test_boundaries.py` fails if
`orchestrator.py`, `mixer.py`, `state.py` or `contracts.py` acquires an ML dependency, and it checks
`sys.modules` after importing the orchestration layer rather than trusting the source.

---

## Two things to know before judging it

**It is not real time yet.** 78.4 ms/frame on a T4 against a 40 ms budget, down from 124.7 once the
CPU half of a frame was overlapped with the GPU half. The per-stage split says where the rest goes —
VAE decode 57.8 ms, and the U-Net only 12.3 ms — so the
next work is a faster decode and moving blending off the critical path, not a bigger GPU. MuseTalk's
paper reports 30+ fps; that is a different card, and it does not include our blending and encoding.
We quote our own number with our own hardware attached.

**There is no authentication.** None, anywhere. The candidate link is not a credential, and the
assistant will read any transcript in the store. A stated development posture —
[SECURITY.md](SECURITY.md) explains why storing real people's faces and voices changes that
calculation.
