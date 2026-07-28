# M0 triage — run 1 failed before touching the GPU

## What the numbers actually say

Run of 2026-07-28 10:44, Tesla T4 15360 MiB, driver 580.82.07, Python 3.12.13,
MuseTalk at commit `0a89dec`.

**No inference happened.** Three fields establish that on their own:

| Field | Value | What it means |
|---|---|---|
| `peak_vram_mib` | **3** | The GPU was never used. A T4 running any talking-head model sits in the thousands. |
| `weights_on_disk` | **96M** | MuseTalk's checkpoints total several GB. Almost nothing downloaded. |
| `inference_cold.exit_code` | **1** | The command failed. Same for warm and both realtime runs. |
| output video | none | The notebook correctly refused to record an fps number. |

**Therefore these are not measurements and must not be quoted as any:**

- `cold_warm_ratio: 2.1` — the ratio between two crashes, one of which crashed slower.
- `identity_prep_s: 0.25` — the difference between two identical failures.
- `inference_warm.seconds: 15.43` — how long it took to fail, not to render.

The notebook's own guard caught this (`"no output video — nothing has been proven
yet; do not record any fps number"`). That guard did its job.

## What *is* a real finding

Two things, and both belong in the model-selection memo rather than the results table:

1. **A free-tier T4 with 15GB was available on first try.** That resolves the hardware
   question the whole spike existed to answer. Whatever runs, runs on this.
2. **Setup fails silently.** `download_weights.sh` exited 0 having fetched 96MB, and
   `pip install` exited 0 in 13 seconds. Both *reported success*. That is a fragility
   finding with teeth — `PROCESS.md` §2.1 has a "setup fragility" criterion, and
   "the install script exits 0 without installing the model" is exactly the evidence
   that criterion is asking for.

## The two suspects

### Suspect 1: weights never downloaded (most likely)

96MB is roughly the small auxiliary files — face parsing, resnet18, configs — and none
of the actual model. The large checkpoints come from Hugging Face and commonly fail for
one of three reasons:

- `git-lfs` not installed, so LFS pointer files land instead of the real weights
- `huggingface_hub` / `gdown` missing, so the fetch step is skipped
- a 403 or rate-limit on one file, which the shell script does not treat as fatal

### Suspect 2: `mmcv` / `mmpose` not actually installed

A 13-second `pip install` for MuseTalk is implausible. It depends on `mmcv` and
`mmpose`, which normally compile or pull large wheels. Colab already ships torch, so
pip may have resolved most of the tree as satisfied while the OpenMMLab packages
silently did not land — and MuseTalk imports them at startup, which would produce
exactly the exit code 1 seen here.

## Run this next — paste into a fresh Colab cell

This does not attempt inference. It answers *which* of the two suspects is real, and
prints the stderr the JSON block did not carry.

```python
import subprocess, os, pathlib

os.chdir('/content/m0/MuseTalk')

def sh(cmd):
    print(f'\n$ {cmd}')
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print((r.stdout or '')[-2500:])
    if r.returncode != 0:
        print('STDERR:', (r.stderr or '')[-2500:])
    return r

print('=' * 70)
print('1. WHAT ACTUALLY LANDED IN models/')
print('=' * 70)
sh('find models -type f -size +1M -exec ls -lh {} \\; 2>/dev/null | head -30')
sh('du -sh models/* 2>/dev/null | sort -h')
# LFS pointer files are ~130 bytes of text starting with "version https://git-lfs"
sh('find models -type f -size -1k -exec head -c 60 {} \\; -exec echo "  <-- {}" \\; 2>/dev/null | head -40')

print('=' * 70)
print('2. CAN THE IMPORTS RESOLVE?')
print('=' * 70)
for mod in ['torch', 'mmcv', 'mmpose', 'mmdet', 'diffusers', 'transformers', 'omegaconf']:
    r = subprocess.run(
        f'python -c "import {mod}; print({mod}.__version__)"',
        shell=True, capture_output=True, text=True,
    )
    status = (r.stdout or '').strip() if r.returncode == 0 else 'MISSING'
    print(f'  {mod:16} {status}')

print('=' * 70)
print('3. THE ACTUAL ERROR (this is the bit the JSON did not carry)')
print('=' * 70)
sh('ls configs/inference/ 2>/dev/null; ls scripts/*.py 2>/dev/null')
sh('python -m scripts.inference --inference_config configs/inference/test.yaml '
   '--result_dir ./results/triage 2>&1 | tail -40')

print('=' * 70)
print('4. WHAT THE README SAYS TO RUN')
print('=' * 70)
sh("grep -n -A12 'inference' README.md | head -60")
```

Paste the whole output back. The `find ... -size -1k` line is the one that settles
suspect 1: if it prints files beginning `version https://git-lfs`, the weights are LFS
pointers and the fix is `apt-get install git-lfs && git lfs pull`.

## Stop conditions still apply

You are roughly 30 minutes into a one-day box, and the failure is in setup rather than
in the model — which is the cheap kind. Run the diagnostic above. If it does not point
at an obvious fix within an hour, the fallbacks in `M0_SPIKE.md` §5 are all still open,
and "MuseTalk's setup script reports success without installing the model" is already a
publishable finding whichever model you end up picking.
