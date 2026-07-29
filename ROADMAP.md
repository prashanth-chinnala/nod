# Roadmap — console, knowledge base, and text chat

Branch: `real-avatar`. This is a plan for work *beyond* the take-home, and it is deliberately
kept off `main` so the assessment repo stays the thing the brief asked for.

**Read the scope note first.** It changes what I recommend doing this week.

---

## 0. Scope note, up front

The full ask — a Tavus-style console managing agents, faces, knowledge bases, tools,
guardrails, and pronunciations, split into an API/web monorepo — is **roughly 14 focused
days** by the estimates in §5. That is not a caveat, it is the plan's headline number.

Two consequences worth deciding on deliberately:

**It should not go on `main` before the submission.** The brief says this is *"deliberately
the lowest-code, highest-judgment assessment we run"* and grades *"boring technology, sharp
execution"*. A half-finished admin console on `main` reads as scope confusion — it trades the
thing being graded for volume that isn't. `main` is submission-ready today; this branch is
where the product work happens.

**Three of these items are genuinely small and demo well.** Text chat is nearly free, the
knowledge base has a real one-day version, and both make the interviewer visibly smarter on
camera. Everything else is product infrastructure that does not change what a reviewer sees.

So the recommendation is a **two-track split**: land §2 (Today) before recording, and treat
§3–§4 as post-submission product work.

---

## Status — updated as this lands

**The ultimate goal, restated so it does not drift:** a working conversational avatar product
that can be demoed end to end — a real face driven by a real voice, configured through a
console, with a knowledge base, and honest numbers throughout. `main` holds the assessment
deliverable; this branch is the product.

| Roadmap item | Est. | State |
|---|---|---|
| Text chat + two-sided transcript | 2 h | **Done.** Verified live — a typed answer produced a follow-up quoting "40,000 corrupted records" |
| Monorepo — `apps/api` + `apps/web`, pnpm, two CI jobs | — | **Done.** 534 API tests green; web builds, typechecks, lints |
| Console shell — tokens, 9 primitives, grouped nav | 1.5 d | **Done** |
| JSON resource store, atomic writes | — | **Done** |
| Knowledge base v1 — retrieval | 1 d | **Done.** BM25 + Chroma Cloud behind one Protocol, 22 tests |
| Pronunciations — lexicon | 2 h | **Done.** 19 tests, including `C++` / `C#` / `.NET` / `Node.js` |
| Agents CRUD | 2 d | **Done** — router, tests, page. Turn-taking params exposed |
| Faces + prep queue | 1.5 d | **Done** against the placeholder renderer. Real prep needs a GPU |
| Knowledge management UI | 2 d | **Done** — incl. the interactive retrieval tester |
| Tools | 2 d | **CRUD done.** Not yet callable mid-turn — see the gap below |
| Guardrails | 1.5 d | **CRUD + `/check` done.** Not yet enforced mid-turn |
| Sessions and transcripts | 1.5 d | **CRUD + page done.** The runtime does not write records yet |
| MuseTalk renderer behind the Protocol | — | **Written, never executed.** 27 GPU-free tests |
| Design and polish pass | 2 d | Partial — primitives and tokens exist, no dedicated pass |

### The gap that matters, stated plainly

**The console configures things the runtime does not yet consult.** Every resource has an
API, tests, and a page; the live conversation currently reads none of them. A knowledge base
can be created and its retrieval tested in the UI, and the interviewer still will not use it
until the wiring below lands. That distinction is easy to lose behind a row of green ticks,
so it gets its own heading.

| Wiring | State |
|---|---|
| Retrieval → the prompt | In progress. Decorator on `SentenceStream`, no orchestrator change |
| Pronunciation → TTS | Module done, decorator written, not yet constructed in `server.py` |
| Guardrails → the turn | Not started. Input check before the LLM, output check before TTS |
| Tools → a call loop | Not started. **The only item that changes the orchestrator** |
| Sessions ← the runtime | Not started. `heard`/`said`/latency exist as telemetry; nothing persists them |
| An agent selected per session | Not started. The socket ignores `agent_id`; config comes from env vars |
| Console overview page at `/` | Not started |
| A real face | **Blocked on a GPU spike run** |

### Decisions changed since the first draft

**Next.js, not Vite.** Asked for, and defensible on a product basis rather than taste: this
grows two surfaces, an internal console where server rendering buys little, and a
candidate-facing interview page reached by a shared link, where first paint and a per-session
route genuinely matter. One framework serving both beats two toolchains.

**Full monorepo, not `console/` alongside.** The earlier draft argued against moving the API
while a submission was pending. That constraint was lifted deliberately — the assessment
deliverable is frozen on `main`, so the restructure costs nothing it was protecting.

**A JSON-file store, not Postgres.** Every resource here is a handful of small documents
edited by one operator. A database would add a service, migrations, and a pool to tune in
exchange for guarantees nothing needs. Writes are atomic via temp-file-plus-rename, because a
half-written JSON file is a permanently broken resource. The limit is written down in
`store.py`: two concurrent operators, or a few thousand rows, and it should be replaced
rather than extended.

---

## 1. Repository shape

**Recommendation: add, do not move.** A full `apps/api` + `apps/web` restructure means
touching every path in `pyproject.toml`, the CI workflow, `PROCESS.md`, `README.md`, and
every import — for zero functional gain, while a submission is pending.

```
nod/
  src/avatar/          # the API. Unchanged. CI green, 252 tests.
  web/                 # the live session client. Unchanged, no build step.
  console/             # NEW — the management UI. Its own toolchain.
    src/
    package.json
    vite.config.ts
  pyproject.toml
```

`console/` has its own `package.json` and never touches the Python build, so the two evolve
independently and CI stays exactly as it is. If a true monorepo is wanted later, the rename
to `apps/` is one mechanical commit — do it when something actually needs the shared
tooling, not before.

**Why the session client stays plain JS.** `web/index.html` is a measuring instrument with
no build step, and the README promises a clean clone reaches a running prototype. Putting it
behind Vite would break that promise to gain nothing. The console is a different kind of
thing and gets a different toolchain.

### Console stack

| Choice | Why this and not the alternative |
|---|---|
| **Vite + React + TypeScript** | Fastest path to a good SPA. Types shared with the API via generated OpenAPI clients |
| **Tailwind + Radix primitives** | Radix gives accessible dialogs, menus, and popovers correctly; Tailwind keeps styling in one place. Not a component kit that owns the look |
| **TanStack Query** | Server state is the whole app. Caching, invalidation and optimistic updates are the bulk of the work, and hand-rolling them is where admin UIs rot |
| **openapi-typescript** | FastAPI already emits OpenAPI. Generating the client means an endpoint change becomes a **type error**, not a runtime surprise |
| **No global state library** | Server state belongs to TanStack Query, form state to the form. Redux here would be state about state |

---

## 2. Today — the parts that pay for themselves before recording

### 2.1 Text chat alongside voice — ~2 hours

Tavus CVI has text chat next to the video, and **most of this already exists.** The server
already accepts a client-supplied transcript:

```python
# server.py — already there
await self._orchestrator.on_end_of_turn(str(message.get("transcript", "")))
```

So a text box that sends `{type: "end_of_turn", transcript: "..."}` drives a full turn with
no server change at all. Work is: an input in `web/index.html`, a transcript pane showing
both sides, and Enter-to-send.

**Why it is worth doing before the Loom, beyond looking good:** it removes STT from the
critical path of the demo. If Deepgram returns nothing on the day, you type a sentence and
the interviewer still visibly reasons over it — the follow-up references what you wrote.
That converts the single most fragile part of the live demo into a fallback you control.

### 2.2 Knowledge base v1 — ~1 day

The interviewer currently asks generically strong questions. Given a job description it
would ask *role-specific* ones, which is a visible, explainable jump in quality.

**Deliberately no embeddings in v1.** For a handful of short documents, BM25-style keyword
retrieval over paragraph chunks is genuinely competitive, needs **no new dependency, no
vector store, and no embedding API call on the critical path** — and the latency budget
(§1.5) has no room for another network hop. Embeddings are §4.3 when the corpus justifies
them.

```
src/avatar/knowledge/
  store.py       # load documents from a directory, chunk on blank lines
  retrieve.py    # BM25 scoring, top-k, character budget
  __init__.py    # build_knowledge(name) registry, mirroring build_llm
```

Wiring: retrieve against the candidate's latest transcript, inject the top chunks into the
system prompt as context. One new boundary, same shape as the others — a `Retriever`
Protocol with a null implementation as the default, so a clean clone still runs.

**Latency cost must be measured, not assumed.** BM25 over a few hundred chunks is
sub-millisecond, but the claim goes in `PROCESS.md` §1.5 with a number next to it or not at
all.

### 2.3 Pronunciations — ~2 hours

The cheapest quality win available. Deepgram Aura mispronounces exactly the words an
engineering interview is full of: *Kubernetes*, *PostgreSQL*, *nginx*, *Kafka*, plus
candidate names.

A dict applied to the text *before* it reaches TTS: `{"nginx": "engine ex"}`. No API
feature, no model change, testable with plain string assertions. It also demonstrates a real
point — that the orchestration layer can improve output quality without touching the model.

---

## 3. The console — the product work

Each entity below is a page, and each is independently useful. Build in this order; the
ordering is by how much the *next* one depends on it.

### 3.1 Agents (the central object) — 2 days

Tavus calls this a Persona. It is the config that everything else attaches to.

```
Agent
  id, name, created_at, updated_at
  system_prompt          text
  llm: { provider, model, temperature }
  voice: { provider, voice_id, speed }
  face_id                -> Face
  knowledge_base_ids     -> [KnowledgeBase]
  tool_ids               -> [Tool]
  guardrail_id           -> Guardrail
  pronunciation_id       -> Pronunciation
  turn_taking: { onset_probability, release_probability, end_of_turn_silence_ms,
                 min_speech_ms, onset_frames }
```

**Exposing turn-taking in the UI is the opinionated call**, and it is the right one: §1.5
shows `end_of_turn_silence_ms` is the largest single term in the latency budget. It is a
product decision — an interview for a senior role wants a longer pause than a screening
call — so it belongs in the hands of whoever designs the interview, not in a config file.

`POST /agents/{id}/sessions` returns a session token; the existing client connects with it.
That is the seam between console and runtime.

### 3.2 Faces — 1.5 days

Upload a reference clip → run `prepare_identity` → cache the artifact. This is §1.2 made
into a product feature, and it needs a **job queue**, because preparation takes seconds to
minutes and must not block a request.

```
Face
  id, name, reference_path, thumbnail_path
  status: queued | preparing | ready | failed
  prepared_artifact_path, enrollment_ms, failure_reason
  created_at
```

UI: drag-and-drop upload, a live status chip, a preview frame, and **`enrollment_ms` shown
in the list** — because that number is architecturally interesting and currently
`NOT YET MEASURED`.

**Blocked on a GPU.** The renderer exists (`renderers/musetalk.py`) but has never executed.
Until the spike runs, this page can be built against the stub renderer, whose
`prepare_identity` returns instantly — which is enough to build and test the queue, the
status transitions, and the UI.

### 3.3 Knowledge bases — 2 days

The v1 from §2.2, given a management surface: create, upload documents, see chunk counts,
and — the part that matters — **test retrieval interactively.** A query box that shows which
chunks would be retrieved and their scores. Without that, a knowledge base is a black box and
nobody can tell a bad retrieval from a bad answer.

```
KnowledgeBase
  id, name, description, chunk_count, total_chars, updated_at
Document
  id, kb_id, filename, content_type, chunk_count, uploaded_at
```

### 3.4 Tools — 2 days

Functions the agent may call mid-interview: `score_answer`, `lookup_candidate_history`,
`flag_for_review`, `end_interview`.

```
Tool
  id, name, description
  parameters_schema   JSON Schema
  implementation: { kind: "http" | "builtin", url, headers, timeout_ms }
  enabled
```

**The hard part is not the CRUD, it is the latency.** A tool call inserts a round trip
*inside* a conversational turn that is already 2.7–5.8s. Two things follow: tools need a
hard timeout with a defined fallback, and the UI must show measured p95 per tool. A tool
that takes 900ms is a product decision, not an implementation detail.

Needs an orchestrator change — the LLM boundary becomes a loop rather than a single stream —
and that is the first item on this list that touches the state machine. Sequence it after
the pages that do not.

### 3.5 Guardrails — 1.5 days

```
Guardrail
  id, name
  banned_topics        [string]
  pii_redaction        bool
  max_answer_chars     int
  refusal_message      text
  on_violation: "refuse" | "redirect" | "end_session"
```

Two enforcement points, and both are needed: on the transcript before it reaches the LLM,
and on generated text before it reaches TTS. Output-side enforcement has a real cost —
checking a sentence before speaking it adds latency to a budget with none to spare — so the
check must be a local string/regex pass, not a model call.

### 3.6 Sessions and transcripts — 1.5 days

The page that makes everything else debuggable: per-session transcript with both sides,
turn-by-turn latency breakdown, barge-in events, the `heard` events, and stale-drop counts.

This is `PROCESS.md` §1.7's observability plan with a UI on it, and it is where the
`heard` telemetry added this week pays off — an empty transcript becomes visible in a list
rather than needing a log grep.

---

## 4. Later

| Item | Why not now |
|---|---|
| Monorepo rename to `apps/` | Mechanical. Do it when shared tooling actually exists |
| Embedding-based retrieval | Only when BM25 is measurably insufficient. Adds a dependency and a network hop to a budget with none spare |
| Multi-tenant auth | Needed before anyone but us uses it; not needed to build any page above |
| WebRTC transport | §3.4 of `PROCESS.md`. Independent of all console work |
| Warm renderer pool | Only matters once faces are real and concurrency exists |

---

## 5. Estimates, honestly

| Item | Estimate | Depends on |
|---|---|---|
| Text chat | 2 h | Nothing — server already accepts it |
| Knowledge base v1 (retrieval only) | 1 d | Nothing |
| Pronunciations | 2 h | Nothing |
| Console shell — routing, layout, design system | 1.5 d | Stack decision |
| Agents CRUD | 2 d | Console shell |
| Faces + job queue | 1.5 d | Console shell. **GPU for real prep** |
| Knowledge base management UI | 2 d | KB v1, console shell |
| Tools | 2 d | Agents. **Touches the orchestrator** |
| Guardrails | 1.5 d | Agents |
| Sessions and transcripts | 1.5 d | Console shell |
| Design and polish pass | 2 d | Everything above |
| **Total** | **~14 d** | |

The first three rows total **~1.5 days** and are the only ones that change what a reviewer
sees. The remaining ~12.5 days build a product.

---

## 6. Design direction for the console

"Beautiful" needs to mean something checkable, so:

**It is an instrument, not a marketing page.** Dashboards are scanned and operated, not read
top to bottom. Summary before detail; state encoded in form as well as in text, so what needs
attention reads at a glance — a `failed` face and a `ready` face must be distinguishable
without reading the word.

**Latency is a first-class visual.** This product's defining characteristic is a latency
budget it does not currently meet. Every list that can show a p95 should, and the turn
breakdown should be a waterfall — the same one `web/index.html` already draws — not a table
of numbers. Show the sub-second target as a line on the chart, because the gap is the story.

**Semantic colour is separate from brand accent.** good / warning / critical carry meaning;
the accent carries identity. Conflating them is why most admin UIs become unreadable at
density.

**Type does the work.** Tabular numerals wherever figures align in columns — a latency table
with proportional digits is unreadable. One display face, one text face, one mono for
transcripts and IDs.

**Empty states are the most-seen screens.** Every page starts empty. Each needs a real one:
what this is, why it matters, and the single action that fills it.

---

## 7. What I would do next, concretely

1. **Text chat** (2 h) — de-risks the Loom, then record.
2. **Knowledge base v1 + pronunciations** (1.5 d) — visible quality jump, no new dependency.
3. **Then** the console, starting with shell + Agents, once the submission is in.

Item 1 is the one I would not skip. It turns the most fragile part of the live demo into
something you control, for two hours of work.

---

## Running the SFU locally

```bash
docker run -d --name nod-livekit \
  -p 7880:7880 -p 7881:7881 -p 7882:7882/udp \
  -e LIVEKIT_KEYS="devkey: secret_that_is_long_enough_for_livekit" \
  livekit/livekit-server:latest --dev --bind 0.0.0.0 --node-ip 127.0.0.1
```

**`--node-ip 127.0.0.1` is not optional and cost an hour to find.** Without it the SFU
advertises Docker's bridge address (`172.17.0.2`) as its ICE candidate. Signalling then
succeeds over the mapped WebSocket port, the browser reports *"connected to LiveKit Server"*,
and the peer connection immediately fails — presenting as `connecting → disconnected` with
`DisconnectReason 0` (unknown) and, in the room, repeated joins for the same identity.

Every symptom points at the application: the token is valid, the room is right, both
participants appear in the server log, and the agent publishes both tracks. Nothing points at
ICE. The tell is that the Python SDK works from the host while the browser does not — the
Python client falls back to TCP on 7881, and browsers are stricter.

### When the call does not establish

Two independent causes, both of which present as "no video, chip says websocket":

1. **The SFU's ICE candidate** — see `--node-ip 127.0.0.1` above.
2. **A stale Next bundle.** This one leaves no trace: source is correct, `typecheck` and
   `lint` pass, the served chunks contain the new code, and the browser still runs an older
   module. `rm -rf apps/web/.next` and restart `pnpm dev`, then hard-reload the tab.

Distinguishing them takes one query — if the LiveKit room log shows only `avatar-agent` and no
`candidate-*` participant, the browser never joined, which is cause 2. If both joined and the
connection went `connecting → disconnected`, it is cause 1.
