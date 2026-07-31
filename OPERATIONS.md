# Operations

Running `nod` for real: on a GPU host, with the media plane, and with the numbers you should expect
so that "it seems slow" can be checked rather than felt.

---

## 1. What runs where

| Process | Port | Needs |
|---|---|---|
| `avatar.server` (FastAPI + WebSocket) | 8000 | Python 3.11+; a GPU only for `AVATAR_RENDERER=musetalk` |
| `apps/web` (Next.js console + interview room) | 3000 | Node 22, pnpm |
| `apps/assistant` (LangGraph) | 8100 | its own venv; reads the same store |
| LiveKit SFU + egress + Redis | 7880, UDP 51000–51050 | Docker; only for WebRTC and recording |
| Postgres | 5432 | optional — the default store is JSON files |

The API is the product. Everything else can be absent and an interview still works.

---

## 2. On a GPU host

Verified on a Lightning.ai Studio: Tesla T4, 15 GB, 4 vCPU, Ubuntu 24.04.

```bash
git clone -b real-avatar <repo> && cd nod/apps/api
sudo apt-get install -y ffmpeg python3.12-venv
./scripts/setup_musetalk.sh
```

`setup_musetalk.sh` creates its own venv, clones MuseTalk, installs a pinned dependency set, fetches
3.7 GB of verified weights, and then **loads every model and reports which device it got**. That
last step is deliberate: importing successfully proves nothing, and a silent fall back to CPU is the
difference between a demo and a slideshow.

Then serve, with the two settings that matter:

```bash
AVATAR_RENDERER=musetalk \
AVATAR_FPS=8 \
AVATAR_DATA_DIR=/abs/path/data \
AVATAR_MEDIA_DIR=/abs/path/media \
  uvicorn avatar.server:app --host 0.0.0.0 --port 8000
```

**`AVATAR_FPS` is not cosmetic.** A renderer that misses its target does not degrade gracefully — it
fails completely, because the mixer discards any frame that misses its slot. At 25 fps on hardware
that sustains 8, every frame is late and the candidate watches the placeholder while the interviewer
talks. Set it to what the hardware measures. Run `scripts/bench_renderer.py` to find out.

**Set both directory variables absolutely.** They default to relative paths, so two processes started
from different directories will silently disagree about where the data is. This has cost three
separate debugging sessions in this repository.

### The dependency window, and why it is narrow

Pinned because the edges are sharp, not out of caution:

| | |
|---|---|
| `numpy>=2.1,<2.5` | ≥2 because `scipy` and `diffusers` use `np.long`, which only exists in numpy 2; ≤2.4 because numba rejects 2.5 |
| `opencv-python>=4.12,<5` | 4.9 is compiled against the numpy 1.x ABI and aborts on import against numpy 2 |
| `transformers==4.39.2`, `diffusers==0.30.2`, `huggingface_hub==0.30.2` | MuseTalk's own pins |

Two of upstream's requirements are deliberately **dropped**: `tensorflow` and `tensorboard` are
never imported anywhere in MuseTalk — checked with grep across every `.py` — and `tensorflow==2.12`
has no arm64 macOS wheel, so on a Mac they are a hard install failure in service of dead weight.
`mmcv` and `mmpose` are dropped too; see [MODELS.md](MODELS.md) §5.

### Reaching it from a laptop

A Lightning Studio proxies HTTP ports, but not arbitrary UDP — so a self-hosted SFU there cannot
reach a browser elsewhere. For development, forward both ports instead:

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=20 \
    -L 127.0.0.1:8000:localhost:8000 -L 127.0.0.1:3000:localhost:3000 <host>
```

Two things learned the hard way:

- **`ExitOnForwardFailure` and the explicit `127.0.0.1:` prefix are both load-bearing.** Without
  them, if something local already holds port 8000, ssh binds only `[::1]` and the tunnel *appears*
  to work — while `localhost` resolves to a mixture of the local process and the remote one. Half the
  probes hit the wrong machine and the results are incoherent.
- **`ServerAliveInterval` matters** because the tunnel dies on an idle timeout otherwise, and the
  failure looks like the API being down.

**Which machine am I talking to?** `curl -s localhost:8000/config` and read `store`: `Store` means
JSON files, `PostgresStore` means Postgres. If those differ between your two hosts, that field
identifies which one answered.

---

## 3. Numbers to expect

From [MEASUREMENTS.md](MEASUREMENTS.md). If what you see is far from these, something is wrong.

| | Tesla T4 |
|---|---|
| Model load, once per process | 27 s |
| Enrollment, 550-frame reference | 126 s |
| Session start, identity already cached | **1.5 s** to first frame |
| Session start, cold identity | 70 – 150 s |
| Steady-state render | 114.7 ms/frame ≈ 8.7 fps |
| Frame size at 512 px tall | 29 – 32 KB ≈ 2 Mbps at 8 fps |
| `avatar_first_frame`, warm turn | 2.3 – 3.0 s |
| `frames_discarded`, per turn | 33 – 79 |

**The cold path is the worst of these**, and it is paid by whoever arrives first. Warming the models
and preparing the attached faces at startup is the obvious fix and is not yet done —
[ROADMAP.md](ROADMAP.md).

**Enrollment is not linear in frames.** A fixed cost dominates below ~250 frames, so a 550-frame
reference costs only 1.5× a 150-frame one while giving a 22 s loop instead of 6 s. Prefer longer
references.

### Reading a live session

Every session emits its own telemetry to the server log as JSON lines:

```
{"event":"latency","stage":"llm_ttft","ms":1947,"epoch":8}
{"event":"latency","stage":"tts_first_audio","ms":450,"epoch":8}
{"event":"latency","stage":"avatar_first_frame","ms":3109,"epoch":8}
{"event":"counter","counter":"frames_discarded","amount":39}
{"event":"heard","text":"[1504ms of speech, no transcript]","transcribed":false}
```

- `transcribed: false` means the audio arrived but no transcript did. The interviewer still asks a
  plan question, so the interview *appears* to work while ignoring what was said — check the STT
  credential and that endpointing is configured.
- `frames_discarded` rising with `avatar_first_frame` means the turn's audio finishes before the
  renderer catches up. That is a first-frame-latency problem, not a throughput one.

---

## 4. The media plane, for recording

Recording needs a recorder, which the SFU binary does not include:

```bash
docker compose --env-file .env.development up -d      # SFU + egress + redis
```

`--env-file` is not optional — the SFU reads its keys from there — and **`LIVEKIT_NODE_IP` must be a
LAN address, not loopback**: to a container, `127.0.0.1` is itself, and the failure is a silent
"Start signal not received". Both mistakes fail quietly.

Redis here is a **pub/sub job bus** between the SFU and the egress worker, not a database. Nothing
is persisted in it.

Recording is configured as `RoomEgress` on `CreateRoomRequest`, so **the room must exist before the
first participant joins** — `CreateRoom` does not retrofit egress onto a room that already exists,
and a browser wins that race by about a second. Room creation therefore happens in the token
endpoint.

---

## 5. When something is wrong

| Symptom | Likely cause |
|---|---|
| Room stuck on *Connecting…* | Next compiling the route on first request; reload once |
| Interviewer visible but silent | agent's `voice_provider` is `tone` — a sine wave, not speech |
| Silent, and no "Enable sound" banner | check `buffersStarted` in the browser; the socket path reports a suspended AudioContext now, but the banner needs a click |
| Speech heard, never transcribed | `transcribed: false` in the log; STT credential or endpointing |
| Placeholder face instead of the persona | the renderer offered no idle loop — an identity prepared before that feature, or `AVATAR_RENDERER=stub` |
| First session after a restart hangs for a minute | cold path: model load plus enrollment. Expected; the next session is warm |
| Face switch is slow | the identity cache holds two entries, so switching between more faces re-prepares |
| `enrollment_ms: 0` | enrollment ran against the stub. Check `AVATAR_RENDERER` |

### Two failure modes worth naming

**An orphaned GPU process holds the whole card.** An interrupted SSH command leaves the remote
Python running, and the next run fails with an OOM that looks like a capacity problem. Check
`nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv` before believing it.
Start long remote work with `setsid nohup … < /dev/null &`.

**`ffmpeg` eats a heredoc.** It reads stdin for interactive commands, so an `ffmpeg` call inside a
`ssh host bash -s <<'EOF'` block silently consumes the rest of the script — the symptom is later
lines executing with their first characters missing. Always `ffmpeg -nostdin`.
