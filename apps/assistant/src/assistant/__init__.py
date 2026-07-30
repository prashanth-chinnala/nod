"""
The console assistant.

**Why this file checks for a package without importing it.** `avatar` is the sibling runtime at
../api and cannot be declared as a dependency: this repo has no workspace tool, and a bare
`avatar` requirement resolves to an unrelated distribution of that name on PyPI. So the
dependency is real but unenforced by packaging, and the bare failure -- a `ModuleNotFoundError`
for `avatar.store` -- reads as a broken install of *this* package rather than a missing install
of another one.

**The check must not import it.** `avatar.store` chooses its backend from `AVATAR_STORE` at
import time, and this module runs before `server.py`'s body -- so an `import avatar.store` here
builds the store *before* `load_env()` has read `.env.development`. The first version of this
file did exactly that and silently put the whole service back on the file store: every tool
returned zero rows, and the assistant reported "no sessions have been scored" about a pipeline
with a complete scorecard in Postgres. It was the same ordering bug this package had already
been fixed for once, reintroduced by the check meant to make a different failure legible.

`find_spec` answers "is it installed" without executing it, which is the only form of this check
that is safe here.
"""

from __future__ import annotations

import importlib.util


def _require_avatar() -> None:
    if importlib.util.find_spec("avatar") is not None:
        return
    raise ModuleNotFoundError(
        "the assistant needs the `avatar` runtime package, which lives beside it at "
        "apps/api and is not installable from PyPI -- that name belongs to an unrelated "
        "project. Install the sibling first:\n"
        "    pip install -e apps/api\n"
        "    pip install -e apps/assistant"
    )


_require_avatar()
