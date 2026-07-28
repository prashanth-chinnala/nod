# End-to-end test protocol

Also the shot list for the Loom. Six tests; each names what to do, exactly what to watch,
and — the part that matters — **what it proves and what it does not**.

Everything on the page is measured. The page at `/mockup` is the one with invented
numbers; keep them straight when recording.

## Before you start

```bash
cd ~/nod
.venv/bin/python -m uvicorn avatar.server:app          # no env prefix needed
curl -s localhost:8000/config | python3 -m json.tool   # confirm what resolved
```

`/config` must show `llm: openai`, `tts: deepgram`, `stt: deepgram`. If any says
`scripted`, `tone`, or `none`, `.env` was not picked up and every test below is measuring
placeholders.

Open **<http://127.0.0.1:8000>**. Use headphones for tests 4–6 — without them the
avatar's own voice reaches the microphone and it interrupts itself in a loop.

## The panel, left to right

| Where | Reads | Means |
|---|---|---|
| Badge over the canvas | `idle` / `listening` / `thinking` / `speaking` / `cancelling` | Server state, pushed live — not polled |
| `first frame (measured)` | ms | Turn start → first rendered frame handed to the mixer |
| `End-to-end, to paint` | ms | Turn start → **this page** finished drawing it. Always larger |
| `Client fps` | ~25 | Frames actually drawn here, not what the server claims |
| `Frames repeated` | count | Renderer fell behind real time. The signal an fps average hides |
| `Frames discarded` | count | Rendered frames thrown away by a barge-in |
| `Turn epoch` | integer | Increments on every turn **and** every cancellation |
| `Buffer depth` | frames | Rendered frames queued ahead of playback |
| `Audio acked` | ms | Audio this page confirms it **played**. Drives history truncation |
| `Speech probability` | 0.00–1.00 | The server's VAD on your mic. Turns cyan in speech |
| `Mic uploaded` | KB | Proof your audio is leaving the browser |
| Waterfall | LLM TTFT / TTS / First frame | Server-side stage timings for the last turn |
| Log | timestamped lines | State changes, stale drops, flushes, errors |

---

## Test 1 — the track is continuous

**Do:** press **Start session**. Watch for 10 seconds and touch nothing.

**Watch:** badge `idle`. The rectangle brightens and dims slowly — that is the idle loop
breathing. `Client fps` settles near **25**. `Received` climbs steadily. `Turn epoch` stays
**0**.

**Proves:** the video track opens once at session start and carries frames in *every*
state, including idle. A track that only ran while speaking would stall between turns, and
a stalled track is more visible than a dropped frame — it also corrupts the receiver's
jitter estimate, so the recovery is worse than the stall.

**Does not prove:** anything about the model. There is no model.

---

## Test 2 — one full turn, by button

**Do:** **Starts speaking** → **Stops speaking**. Wait.

**Watch, in order:**

1. Badge → `listening`
2. Badge → `thinking`, `Turn epoch` → 1
3. **~2–4 second pause.** This is real and it is the finding, not a bug — see Test 3
4. Badge → `speaking`, a voice speaks a genuine interview question
5. The mouth bar opens and closes **with the audio**
6. `first frame` and `End-to-end, to paint` fill in; the waterfall draws three segments
7. `Audio acked` climbs as it plays
8. Badge → `idle`

**Proves:** the whole chain — turn detection, LLM, TTS, renderer, transport, playback
acknowledgement. The mouth tracking the audio is the audio-to-video mechanism working; the
mouth height is computed from the RMS of each frame's own 40ms of audio.

**Does not prove:** lip-sync of a *face*. Five rectangles is not a talking head.

**Say to camera:** the question it asked is not canned. Answer it in Test 4 and it follows
up on what you actually said.

---

## Test 3 — read the waterfall out loud

**Do:** nothing. Look at the numbers from Test 2.

**Watch:** roughly `LLM TTFT ~1900–3200ms`, `TTS ~950–1300ms`, `First frame` the sum plus a
little. `End-to-end, to paint` slightly above `first frame`.

**Proves the central architectural claim.** Add the 700ms end-of-turn window and a full
turn is **3.7–5.4s** against a sub-second target. **Not one of the three dominant terms is
the renderer.** A perfect zero-latency talking-head model would still leave ~3.4s.

The gap between `first frame` and `End-to-end, to paint` is encode + socket + decode +
paint — the part a server-side measurement cannot see, which is why this page reports its
paint back rather than the server stopping its own clock early.

**Say to camera:** "more GPU" does not fix this, and that is measured rather than asserted.
The levers are a paid low-latency LLM, Aura's WebSocket interface (351ms flat, verified),
and the 700ms window — which is a policy choice no hardware improves.

---

## Test 4 — talk to it

**Do:** tick **stream mic** (grant access). `Mic uploaded` should start climbing and
`Speech probability` should move when you speak. Then say, out loud:

> "We shipped a queue-backed ingest that assumed ordering we never actually had."

Stop and stay quiet.

**Watch:** `Speech probability` rises past ~0.6 and turns cyan → badge `listening` within
~100ms. After **700ms of silence**, badge → `thinking`. Then it answers — and the question
**references your ordering problem**.

**Proves:** Deepgram transcribed you, the transcript reached the LLM as history, and the
turn boundary was decided locally. The follow-up referencing your words is the proof the
transcript arrived — a canned question could not do that.

**Say to camera:** the 700ms is deliberate and it is the largest term in the budget. It is
a conversational judgment, not a technical one: too short and it answers into your thinking
pause, too long and it feels sluggish. No amount of hardware changes it.

**If nothing happens:** check `Mic uploaded` is climbing. If it is 0 KB, mic permission was
denied — check the log for `mic-denied`.

---

## Test 5 — interrupt it (the highest-value shot)

**Do:** ask it something, and **while it is still speaking**, talk over it.

**Watch, and this is the money shot:**

1. Voice stops **immediately**
2. Badge → `cancelling` → `listening`
3. **`Turn epoch` increments**
4. Log fills with amber **`stale-drop`** lines
5. Log shows **`flush-audio — stopped N scheduled buffer(s)`**
6. `Frames discarded` jumps
7. The rectangle returns to breathing — no frozen mid-sentence mouth

**Proves the graded requirement**, and note *how*: the epoch incrementing while stale
frames are dropped is **evidence**. "It reacted immediately" is an assertion; an integer
changing and artifacts being provably discarded is not. Cancellation is an integer write,
not a task kill — a GPU pass in flight still returns frames, and they die at the consumer.

The `flush-audio` line matters on its own: a server-side flush alone would leave *this
page's* buffer playing a sentence the avatar had abandoned. That reads as a laggy
interruption even though the state machine reacted in microseconds.

**Then:** ask a follow-up. It will not pretend you heard the end of the interrupted
question. History was truncated to what this page **acknowledged playing** — `Audio acked`,
not what was sent. The difference is the buffer a barge-in throws away.

---

## Test 6 — the false positive

**Do:** with the mic on, cough once, or knock the desk. Then stay quiet.

**Watch:** `Speech probability` spikes. The badge should **not** move to `thinking`, and
`Turn epoch` should **not** increment.

**Proves:** onset needs a high probability sustained over 3 frames (~96ms), and speech
under 200ms total is *retracted* rather than delivered as an empty turn.

**Caveat to state plainly:** the default detector is an **energy gate**, not a voice
activity detector — it cannot tell a door from a voice, only loud from quiet. A loud enough
cough *will* trigger it. `AVATAR_VAD=silero` is the real detector, and it has never been
executed.

---

## Headless equivalent

```bash
.venv/bin/python scripts/smoke_session.py     # 17 assertions
```

Asserts what watching cannot: that stale-epoch artifacts were dropped from telemetry
rather than merely looking right, that pts are strictly monotonic, and that end-to-end was
measured to paint rather than to socket write.

---

## What to be careful claiming

| True | Not true |
|---|---|
| Real voice, real transcription, real LLM | There is a face |
| Audio drives the video | It is lip-synced to a person |
| Barge-in verified from telemetry | Interruption latency measured to the ear (server-side only, ~0.4ms) |
| Turn detection is tested policy | The detector under it is a real VAD |
| Latency measured to browser paint | Any of it meets the sub-second target |
