"""
Load `.env` into the process environment, if there is one.

Every knob in this project is an environment variable, which keeps configuration out of
the code and secrets out of git. That only works if something actually reads the file --
before this module existed, `.env` was inert and every run needed
`set -a && . ./.env && set +a` typed in front of it. Forgetting that produced a session
that silently fell back to the placeholder LLM, TTS, and transcriber: no error, just
quietly the wrong system.

Three rules, and the first is the one that matters:

**A real environment variable always wins.** Values already present are never
overwritten, so `AVATAR_TTS=tone uvicorn ...` overrides whatever the file says. Config
files that silently beat explicit flags are a bad surprise, and CI/production set real
variables and must not have them clobbered by a stray file.

**A missing file is not an error.** A clean clone has no `.env` and must still run --
every default is a working no-credential one.

**No dependency.** `python-dotenv` would work; twenty lines that handle exactly the
`KEY=value` subset this project writes is a smaller thing to own than a package, and this
loader is deliberately not a general implementation (see `parse` for what it skips).
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILENAME = ".env"

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


def find_env_file(start: Path | None = None) -> Path | None:
    """Nearest `.env` at or above `start`, within `SEARCH_DEPTH`."""
    here = (start or Path(__file__).resolve()).parent
    for candidate in [here, *here.parents][: SEARCH_DEPTH + 1]:
        path = candidate / ENV_FILENAME
        if path.is_file():
            return path
    return None


def load_env(path: Path | None = None) -> dict[str, str]:
    """
    Fill unset variables from `.env`. Returns what was actually applied.

    Called once at import of `avatar.server`. Returning the applied subset rather than
    everything parsed makes it possible to log which knobs came from the file without
    logging the ones that came from the real environment -- and without logging values,
    which are mostly secrets.
    """
    target = path or find_env_file()
    if target is None:
        return {}
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        # An unreadable .env is a permissions problem worth surviving, not crashing on.
        return {}

    applied: dict[str, str] = {}
    for key, value in parse(text).items():
        if key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied
