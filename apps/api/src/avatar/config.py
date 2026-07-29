"""
Load the env files into the process environment, if there are any.

Every knob in this project is an environment variable, which keeps configuration out of
the code and secrets out of git. That only works if something actually reads the file --
before this module existed, `.env` was inert and every run needed
`set -a && . ./.env && set +a` typed in front of it. Forgetting that produced a session
that silently fell back to the placeholder LLM, TTS, and transcriber: no error, just
quietly the wrong system.

Three rules, and the first is the one that matters:

**A real environment variable always wins.** Values already present are never
overwritten, so `AVATAR_TTS=tone uvicorn ...` overrides whatever the files say. Config
files that silently beat explicit flags are a bad surprise, and CI/production set real
variables and must not have them clobbered by a stray file.

**A missing file is not an error.** A clean clone has none of these and must still run --
every default is a working no-credential one.

**No dependency.** `python-dotenv` would work; twenty lines that handle exactly the
`KEY=value` subset this project writes is a smaller thing to own than a package, and this
loader is deliberately not a general implementation (see `parse` for what it skips).
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILENAMES: tuple[str, ...] = (".env.development", ".env.local", ".env")
"""
Candidate files, in **descending precedence**.

Loaded in this order, and because `load_env` only fills variables that are *unset*, the
first file to define a key wins. That gives layering rather than replacement: keep shared
defaults in `.env`, and let `.env.development` override just the handful you are changing,
without duplicating the rest.

`.env.local` sits between them by the usual convention -- machine-specific overrides that
are not tied to an environment name.

All three are gitignored, with no exemption -- there is deliberately no committed
`.env.example`, since a tracked template is one `git add -f` from being a tracked key.
README's Configuration table is the documentation of what these files may contain.
"""

ENV_FILE_OVERRIDE = "AVATAR_ENV_FILE"
"""
Point at one specific file and skip the search entirely.

The escape hatch for anything the search cannot reach -- a mounted secret, a path outside
the repo, a per-test fixture. Set it and `ENV_FILENAMES` is not consulted at all, because a
caller who names a file explicitly does not want a surprise merge from a neighbouring one.
"""

SEARCH_DEPTH = 4
"""
How far up from this file to look for `.env`.

`src/avatar/config.py` to the repo root is three levels; a fourth covers an editable
install whose layout differs. Bounded rather than walking to `/`, because silently picking
up a `.env` from an unrelated parent directory would be worse than not finding one.
"""


def parse(text: str) -> dict[str, str]:
    """
    Parse the `KEY=value` subset this project writes.

    Handles blank lines, `#` comments, `export KEY=value`, and surrounding single or
    double quotes. Deliberately does *not* handle multi-line values, variable
    interpolation, or escape sequences -- a loader that half-implements shell quoting is
    worse than one with an obvious boundary, because the failure is a subtly wrong secret
    rather than a parse error.
    """
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def find_env_files(start: Path | None = None) -> list[Path]:
    """
    Every candidate file that exists, in precedence order.

    Searches one directory at a time from `start` upwards, and **stops at the first
    directory that contains any candidate**. Without that stop, a `.env` in a parent
    directory would silently layer under a `.env.development` in the repo -- which reads
    as a phantom value coming from nowhere.
    """
    here = (start or Path(__file__).resolve()).parent
    for candidate in [here, *here.parents][: SEARCH_DEPTH + 1]:
        found = [candidate / name for name in ENV_FILENAMES if (candidate / name).is_file()]
        if found:
            return found
    return []


def load_env(path: Path | None = None) -> dict[str, str]:
    """
    Fill unset variables from the env files. Returns what was actually applied.

    Called once at import of `avatar.server`. Returns the applied subset rather than
    everything parsed, so a caller can report which knobs came from a file without
    reporting the ones that came from the real environment -- and without touching values,
    which are mostly secrets.

    Precedence, highest first: a real environment variable, then `AVATAR_ENV_FILE` if set,
    then `.env.development`, `.env.local`, `.env`.
    """
    if path is not None:
        targets = [path]
    elif os.environ.get(ENV_FILE_OVERRIDE):
        named = Path(os.environ[ENV_FILE_OVERRIDE])
        targets = [named] if named.is_file() else []
    else:
        targets = find_env_files()

    applied: dict[str, str] = {}
    for target in targets:
        try:
            text = target.read_text(encoding="utf-8")
        except OSError:
            # An unreadable file is a permissions problem worth surviving, not crashing on.
            continue
        for key, value in parse(text).items():
            # Already set means either a real environment variable or a higher-precedence
            # file got there first. Either way it wins.
            if key not in os.environ:
                os.environ[key] = value
                applied[key] = value
    return applied


def loaded_files(start: Path | None = None) -> list[str]:
    """Which files `load_env` would read, for reporting. Names, never contents."""
    if os.environ.get(ENV_FILE_OVERRIDE):
        return [os.environ[ENV_FILE_OVERRIDE]]
    return [str(p) for p in find_env_files(start)]
