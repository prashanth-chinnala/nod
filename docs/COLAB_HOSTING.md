# Getting a real avatar you can talk to, on Colab

The goal: a face that speaks, driven by your voice, in your browser. Colab has the GPU;
your Mac has the microphone. This is how the two meet.

Four steps. Only step 2 is new work, and it is worth doing **today, before M0 succeeds**.

---

## Step 0 — push the repo (yours, ~2 minutes)

Colab clones the code from GitHub, so it has to be there. The remote is already configured
and **empty** — nothing has been pushed.

```bash
cd ~/nod
git push -u origin main        # 21 commits
```

If it asks for a password, GitHub wants a **personal access token**, not your account
password: <https://github.com/settings/tokens> → *Generate new token (classic)* → tick
`repo` → use the token as the password.

**Before you push, sanity-check that no secret is going up.** It should print nothing:

```bash
git ls-files | grep -E '^\.env|secrets'
```

Every `.env*` is gitignored with no exemption — there is deliberately no committed
`.env.example` template — and I verified nothing is tracked. This is still the one command
worth running yourself rather than taking my word for.

This step is on the critical path regardless: §6 of the brief requires a **public GitHub
repo**, so it is a deliverable, not overhead.

---

## Step 1 — verify the hosting path, with the stub (~10 minutes)

Open **`notebooks/run_on_colab.ipynb`** in Colab and run it. You will get a
`https://<random>.trycloudflare.com` URL. Open that on your Mac, tick *stream mic*, and
talk. Real voice, real transcription, real LLM — running on Colab, driven from your
kitchen table.

**Do this before M0.** It proves four things that have nothing to do with the model and
would each sink the demo on their own:

| | Why it can fail |
|---|---|
| A tunnel proxies WebSockets | Some do not. Both video down and microphone up ride one. |
| The page gets a secure context | Browsers refuse microphone access over plain HTTP. |
| The client upgrades to `wss://` | Mixed content is blocked; the page has to switch scheme with the tunnel. |
| Colab's runtime stays up long enough | It reclaims idle runtimes, taking the tunnel with it. |

Finding out that a tunnel breaks WebSockets *after* two days on the model would be an
expensive order to learn it in. And if this step fails, the answer is a small rented GPU
box rather than more Colab debugging — better to know now.

Secrets go in **Colab Secrets** (sidebar, key icon), not in a cell. Anything typed into a
cell is saved inside the notebook file.

---

## Step 2 — M0: make the model actually render (yours, ~1 hour)

`notebooks/m0_musetalk_v2.ipynb`, diagnosis in `docs/M0_SPIKE.md`. Three fixes over run 1,
one of which was my bug (a missing `--version v15`).

The bar is one rendered `.mp4` plus real numbers. The notebook refuses to print an fps
figure when no output file exists, and it now **stops** if any checkpoint is missing rather
than letting inference produce a timing that measures a crash.

---

## Step 3 — M2: wire it in (mine, a few hours once step 2 lands)

`renderers/musetalk.py` implementing the `TalkingHeadRenderer` Protocol that already
exists. Face detection, parsing, and latent encoding of the reference frames go in
`prepare_identity`, which is explicitly allowed to be slow because it runs once per
persona. `push_audio` and `frames` handle the per-turn streaming.

Then, in cell 3 of `run_on_colab.ipynb`:

```python
'AVATAR_RENDERER=stub',   ->   'AVATAR_RENDERER=musetalk',
```

Nothing else in that notebook changes. Nothing in the server, the orchestrator, the
transport, the client, or the 199 tests changes either. **That one-line swap is the whole
argument for the boundary** — and it is worth demonstrating live on the Loom rather than
asserting.

You will also need a **reference video**: a few seconds of a face, roughly front-on, ideally
25fps. MuseTalk ships samples under `data/`, which is the fastest path to seeing it work.
Your own face is better for the recording.

---

## What Colab will and will not give you

**Will:** a real talking face, driven by your real voice, in your browser.

**Will not:** a fast one, or a stable one.

- **Slower.** Your audio crosses to Colab's region and back, twice per turn, on top of the
  measured **3.7–5.4s** in §3.3.3. Measure it through the tunnel and compare — and say so
  on the Loom. A demo recorded through a tunnel is not measuring what a local one measures,
  and quietly comparing the two would be the dishonest move.
- **Ephemeral.** Close the tab or idle out and everything is gone: packages, weights,
  tunnel URL. Cell 6 keeps it awake; it does not make it permanent.
- **Not the production shape.** One session, one container, weights cold-loaded at start.
  §1.4 argues that cold-loading per session is exactly the cost a real deployment cannot
  pay — a warm pool is M7, and Colab is the opposite of one.

### The alternative worth pricing

A small rented GPU (Lambda, RunPod) at roughly **$0.50/hour** gives you the same thing
without the ephemerality, in one place, with a stable URL. For the few hours it takes to
record a Loom that is a couple of dollars, and it removes three of the four failure modes
above. Colab is the right answer for M0's throughput measurement. For the recording, rented
is the calmer choice.

---

## Ordering, and one honest caveat

**Step 0 → Step 1 → Step 2 → Step 3.** Step 1 is the cheap de-risking move and needs
nothing from M0.

The caveat, because it should inform how much time this gets: §3.3.3 measured a full turn
at 3.7–5.4s, and **none of the three dominant terms is the renderer**. A perfect
zero-latency model still leaves ~3.4s. M0 through M3 buys you a **face** — which the brief
explicitly puts out of scope for fidelity, and which is genuinely worth having on camera.
It does not buy you a fast conversation. Spend accordingly.
