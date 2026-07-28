# M0, step by step

For someone who has not used Google Colab before. If you have, read
[`M0_SPIKE.md`](M0_SPIKE.md) instead — it is the same job with less hand-holding.

You are trying to answer **one question**: does a talking-head model run on a free
Colab GPU, and how fast? You are not trying to make it look good, integrate it, or
finish anything. A rendered video file plus a page of notes is a complete success.

**Budget one day.** If it is not working after four hours, stop — that is a result too,
and there are cheap fallbacks.

---

## What you are about to use

**Google Colab** is a free Jupyter notebook that runs on Google's machines, including
a GPU. You get a temporary Linux computer in a browser tab. Two things about it will
bite you if you don't know them up front:

1. **It is temporary.** Close the tab, or leave it idle ~90 minutes, and the machine is
   wiped — installed packages, downloaded weights, everything. You start over. This is
   why the notebook times each step: so you know what re-doing it costs.
2. **The free tier is not guaranteed a GPU.** Some days you get a T4. Some days you get
   nothing and have to wait. The notebook's first cell checks, and stops if there isn't
   one, rather than letting you discover it forty minutes later.

**A notebook** is a list of *cells*. A cell is a chunk of code you run on its own by
clicking the ▶ button to its left, or pressing **Shift + Enter**. Output appears
underneath it. You run them **top to bottom, in order** — later cells depend on
variables earlier ones created.

**MuseTalk** is the open-source model you are testing. It takes a video of a face plus
an audio file, and re-renders the mouth to match the audio.

---

## Step 1 — Open Colab and get a GPU

1. Go to **<https://colab.research.google.com>** and sign in with a Google account.
2. **File → Upload notebook**, and upload
   `notebooks/m0_musetalk_spike.ipynb` from this repo.
3. **Runtime → Change runtime type**. Under *Hardware accelerator* choose **T4 GPU**.
   Click **Save**. The page will reconnect.

If T4 GPU is greyed out or unavailable, you have hit free-tier capacity. Wait an hour
and try again, or try [Kaggle](https://www.kaggle.com/code) instead — it offers a
similar free GPU with a weekly quota, and the same notebook works there.

---

## Step 2 — Run the first cell and check what you got

Click the first code cell, press **Shift + Enter**. You want output like:

```json
{
  "gpu": { "name": "Tesla T4", "vram_total": "15360 MiB", "driver": "550.54.15" },
  "python": "3.11.x"
}
```

**Write down the GPU name.** Colab hands out T4, L4, and occasionally A100, and every
number you measure afterwards is meaningless without knowing which one you had. If
you got an L4 or A100 and report those numbers as "a free Colab GPU," that is
misleading in a way the assessment specifically grades.

If it says *No GPU*, go back to Step 1.

---

## Step 3 — Work down the cells, and keep a log

Run each cell in order. **Open a text file or a notes app right now** and keep a running
list as you go. One line per thing that goes wrong, with the time:

```
14:05  pip install printed a load of red about torch versions — it finished anyway, 4 min
14:20  weights download died partway with a 403 — ran the cell again, worked
14:35  inference cell complained about --version, so I deleted that argument
```

**This log is not admin overhead. It is one of the deliverables.** The write-up has a
"setup fragility" criterion sitting empty, and "took two retries and one argument
change, 40 minutes total" is the kind of specific that makes a model-selection memo
credible. Reconstructing it from memory in a week does not work.

### What each stretch of the notebook does

| Cells | What is happening | Roughly how long |
|---|---|---|
| 1–2 | Check the GPU, set up the timing helpers | seconds |
| 3 | Download the MuseTalk code, install its Python packages | 3–8 min |
| 4 | Download the model weights — several GB | 5–20 min, depends on the day |
| 5 | Print what's in the repo so you can see the right command to run | seconds |
| 6–7 | Actually render a video. Twice: cold, then warm | 1–10 min each |
| 8 | Measure what came out — resolution, frames, speed | seconds |
| 9 | Time the identity-preparation step separately | 1–5 min |
| 10 | Print the results block | seconds |

### Things that will look alarming and are fine

- **Walls of red text during `pip install`.** Colab ships its own versions of PyTorch
  and friends, and projects pin different ones. As long as the cell finishes, note it
  and move on. Genuine failure looks like the cell *stopping* with an error.
- **A cell taking several minutes with no output.** Downloads and model loading are
  quiet. The ▶ button turns into a spinning circle while a cell is running.
- **A warning about the runtime restarting after install.** If Colab offers to restart,
  accept — then re-run cells 1 and 2 (they define the helpers) before continuing.

### Things that mean stop and tell me

- Cell 6 fails and cell 5's output doesn't show you an obviously different command
- The weights won't download after two attempts
- Four hours in and no video file has been produced
- It runs, but the last cell says **SLOWER than real time** by a lot — say, above 3×

None of these are your fault and none of them are dead ends. See "If it doesn't work"
below.

---

## Step 4 — Send me the results

The last cell prints a block between two lines of `=` signs. **Select it all, copy it,
and paste it into our session.** It looks like:

```json
{
  "gpu": {"name": "Tesla T4", ...},
  "inference_cold": {"seconds": 84.2, "peak_vram_mib": 6420, "exit_code": 0},
  "inference_warm": {"seconds": 31.7, ...},
  "output": {"resolution": "256x256", "frames": 250, "duration_s": 10.0},
  "effective_fps_warm": 7.9,
  "realtime_ratio": 3.17,
  "setup_notes": [...]
}
```

Paste your hand-written log too, if you kept it outside the notebook.

Then I will put the numbers into the write-up, attributed to the GPU you actually had,
and unblock M2 — wiring the real model in behind the interface that already exists.

---

## Step 5 — The part only you can do

Three questions. They are graded specifically on being *yours*, so I will not draft
answers and you should not want me to:

1. **Which model do you pick, and what is the single decisive criterion?**
2. **What is the strongest argument against your pick?**
3. **What would make you switch?**

You will be in a good position to answer them after doing the above, because you will
have felt the setup fragility rather than read about it. Write a paragraph in your own
voice; it does not need to be polished, and I will not rewrite it.

---

## If it doesn't work

Every fallback here is cheap, and the assessment explicitly says it is grading judgment
and honesty about constraints, not raw performance.

| What happened | What to do |
|---|---|
| No GPU available on Colab | Try Kaggle, or come back in an hour. Not a technical problem. |
| MuseTalk won't install or run at all | Tell me. The second candidate is **Ditto** — worth trying, though its TensorRT requirement fights a temporary machine. |
| It runs but far slower than real time | **That is a valid, reportable result.** Try a smaller resolution first. Then write it down honestly — this is exactly the "what would it take to reach true real-time" question the brief asks. |
| The whole thing is eating days | Stop. Report what you learned. A documented dead end beats an undocumented near-miss, and there are five other deliverables. |

The one thing that would actually cost you marks is inventing a number because the
real one was inconvenient to get. An empty cell that says `NOT YET MEASURED` is
completely fine. A plausible-looking figure nobody measured is the single worst outcome
in this assessment.

---

## Glossary

| Term | What it means here |
|---|---|
| **fps** | Frames per second the model can produce. Real-time conversation needs roughly 25–30. |
| **VRAM** | Memory on the graphics card. A T4 has ~15GB usable. Decides how many conversations one GPU could host at once. |
| **Cold vs warm run** | The first run of anything on a GPU is much slower — memory allocation, kernel compilation, loading weights. The second run is the honest steady-state number. Report both. |
| **Real-time ratio** | Render time ÷ audio duration. Below 1.0 means it can keep up with a live conversation. Above 1.0 means it cannot. |
| **Identity preparation** | One-time work per person — finding the face, encoding the reference frames. Allowed to be slow, because it happens before any conversation starts. Measuring it separately is the point. |
| **Checkpoint / weights** | The trained model file. Gigabytes. Never committed to git. |
| **Inference** | Running the trained model to produce output, as opposed to training it. You are only doing inference. |
