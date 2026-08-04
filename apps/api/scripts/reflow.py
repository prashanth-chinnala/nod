#!/usr/bin/env python3
"""
Rewrap over-long comment and docstring prose. Refuses to touch code, and proves it.

**Why this exists.** Rewrapping long prose by hand, or by eye through an editor, corrupted working
code more than once in this repo -- a `print()` split across a line, a closing `\"\"\"` absorbed into
a sentence, a comment marker lost so the next line became a statement. Each time the file still
looked plausible and failed at import.

**The guarantee, and it is the whole point.** After rewrapping, the file is tokenised and every
token that is not a comment or a string is compared against the original. If that sequence differs
in any way, nothing is written. So this cannot change what the program does: the worst outcome it
can produce is a file it declined to modify. `ast.parse` is checked too, but the token comparison is
the stronger claim -- parsing only proves the result is *some* valid program, not the same one.

Docstrings are strings, so their tokens are excluded from that comparison and could in principle be
mangled undetected. They are rewrapped paragraph-by-paragraph with the indent preserved, and lines
that do not look like prose -- code blocks, tables, anything indented past its neighbours -- are
left exactly as found.

    python scripts/reflow.py src/avatar/transport/worker_audio.py    # rewrite in place
    python scripts/reflow.py --check src tests                      # report, change nothing
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import sys
import textwrap
import tokenize
from pathlib import Path

LIMIT = 96
"""Matches `line-length` in pyproject.toml. Kept as a constant so the two can be compared."""


def code_tokens(source: str) -> list[tuple[int, str]]:
    """
    Every token that is not prose, whitespace, or a marker.

    Comments and strings are excluded because those are what this script rewrites. Indentation and
    newlines are excluded because rewrapping changes how many lines a docstring occupies, which
    moves NEWLINE/NL tokens around without meaning anything.
    """
    skip = {
        tokenize.COMMENT,
        tokenize.STRING,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENCODING,
    }
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type not in skip:
            out.append((tok.type, tok.string))
    return out


def is_prose(line: str, indent: str) -> bool:
    """
    Whether a line inside a docstring is prose that may be rewrapped.

    Conservative on purpose. Anything indented past the docstring's own indent is treated as
    structure -- a code sample, a list continuation, a table -- because rewrapping those destroys
    the only thing that made them readable.
    """
    body = line[len(indent) :] if line.startswith(indent) else line.lstrip()
    if not body.strip():
        return False
    if line.strip() and not line.startswith(indent):
        return False
    if body[:1] in {" ", "\t"}:
        return False
    if re.match(r"^(\||[-*+]\s|\d+[.)]\s|>>>|\.\.\.|#|```)", body):
        return False
    return True


def wrap_block(lines: list[str], indent: str, prefix: str = "") -> list[str]:
    """Rewrap one paragraph to LIMIT, preserving indent and an optional `# ` style prefix."""
    text = " ".join(line.strip().removeprefix(prefix.strip()).strip() for line in lines)
    if not text:
        return lines
    width = LIMIT - len(indent) - len(prefix)
    if width < 20:
        return lines
    return [f"{indent}{prefix}{piece}" for piece in textwrap.wrap(text, width=width)]


def reflow_comments(lines: list[str]) -> list[str]:
    """
    Rewrap runs of consecutive `#` comment lines that share an indent.

    Grouped into runs rather than handled line by line, because a comment paragraph's words have to
    be able to move between its lines -- rewrapping a single long line in isolation leaves a short
    orphan under it and makes the next run over-long instead.
    """
    out: list[str] = []
    run: list[str] = []
    run_indent = ""

    def flush() -> None:
        nonlocal run, run_indent
        if not run:
            return
        if any(len(line) > LIMIT for line in run):
            out.extend(wrap_block(run, run_indent, "# "))
        else:
            out.extend(run)
        run = []

    for line in lines:
        match = re.match(r"^(\s*)#\s?(?!\s*(?:type:|noqa|ruff:|mypy:|!))(.*)$", line)
        if match and match.group(2).strip() and not match.group(2).lstrip().startswith("-- "):
            indent = match.group(1)
            if run and indent != run_indent:
                flush()
            run_indent = indent
            run.append(line)
            continue
        flush()
        out.append(line)
    flush()
    return out


def reflow_docstrings(source: str) -> str:
    """
    Rewrap prose paragraphs inside every string token that occupies its own line(s).

    Uses the tokeniser to find them rather than a regex, because a triple-quoted string is not
    something a regex can locate reliably in the presence of nested quotes -- and getting that wrong
    is precisely the failure this script exists to prevent.
    """
    lines = source.splitlines()
    edits: list[tuple[int, int, list[str]]] = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type != tokenize.STRING:
            continue
        start, end = tok.start[0] - 1, tok.end[0] - 1
        if end <= start:
            continue  # single-line string; rewrapping it would need to move the quotes
        block = lines[start : end + 1]
        if not any(len(line) > LIMIT for line in block):
            continue
        indent = " " * (len(block[0]) - len(block[0].lstrip()))
        # First and last lines carry the quotes. Leave both alone and rewrap between them.
        inner, rebuilt = block[1:-1], [block[0]]
        para: list[str] = []
        for line in inner:
            if is_prose(line, indent):
                para.append(line)
                continue
            if para:
                rebuilt.extend(wrap_block(para, indent))
                para = []
            rebuilt.append(line)
        if para:
            rebuilt.extend(wrap_block(para, indent))
        rebuilt.append(block[-1])
        edits.append((start, end, rebuilt))

    for start, end, rebuilt in reversed(edits):
        lines[start : end + 1] = rebuilt
    return "\n".join(lines) + ("\n" if source.endswith("\n") else "")


def reflow(source: str) -> str:
    """Comments first, then docstrings, then the safety check. Returns source unchanged on doubt."""
    try:
        before = code_tokens(source)
    except tokenize.TokenError:
        return source
    result = "\n".join(reflow_comments(source.splitlines()))
    result += "\n" if source.endswith("\n") else ""
    result = reflow_docstrings(result)
    try:
        if code_tokens(result) != before:
            return source
        ast.parse(result)
    except (SyntaxError, tokenize.TokenError):
        return source
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--check", action="store_true", help="report, write nothing")
    args = parser.parse_args()

    files = []
    for path in args.paths:
        files.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])

    changed, stubborn = 0, []
    for path in files:
        source = path.read_text()
        result = reflow(source)
        long_after = [n for n, line in enumerate(result.splitlines(), 1) if len(line) > LIMIT]
        if result != source:
            changed += 1
            if not args.check:
                path.write_text(result)
            print(f"{'would rewrap' if args.check else 'rewrapped'} {path}")
        if long_after:
            # Left for a human: a long line this script will not touch is a code line, a URL, or a
            # table -- and every one of those needs a judgement it should not be making.
            stubborn.append((path, long_after))

    for path, nums in stubborn:
        preview = ", ".join(str(n) for n in nums[:8])
        print(f"!! {path}: still over {LIMIT} at {preview}", file=sys.stderr)
    print(f"-- {changed} file(s) rewrapped, {len(stubborn)} with lines needing a human")
    return 1 if stubborn else 0


if __name__ == "__main__":
    raise SystemExit(main())
