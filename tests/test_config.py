"""
Environment-file loading, and precedence in particular.

Precedence is the part that fails silently. A config file that quietly beats an explicit
`FOO=bar` on the command line, or a stray `.env` in a parent directory that contributes a
value nobody can account for, both present as "the setting had no effect" — with nothing to
grep for. So every ordering rule gets a test that names what it prevents.

Before this module existed the file was inert: nothing read it, and every run needed
`set -a && . ./.env && set +a` typed in front. Forgetting produced a session that fell back
to every placeholder with no error at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from avatar.config import (
    ENV_FILE_OVERRIDE,
    ENV_FILENAMES,
    find_env_files,
    load_env,
    loaded_files,
    parse,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing here may leak into the real process environment."""
    for key in ("ALPHA", "BETA", "GAMMA", ENV_FILE_OVERRIDE):
        monkeypatch.delenv(key, raising=False)


# -- parsing ---------------------------------------------------------------


def test_parses_the_subset_this_project_writes() -> None:
    parsed = parse(
        "\n".join(
            [
                "# a comment",
                "",
                "ALPHA=one",
                "export BETA=two",
                "GAMMA='three'",
                'DELTA="four"',
                "  EPSILON = five  ",
            ]
        )
    )

    assert parsed == {
        "ALPHA": "one",
        "BETA": "two",
        "GAMMA": "three",
        "DELTA": "four",
        "EPSILON": "five",
    }


def test_lines_without_an_equals_are_skipped_not_guessed() -> None:
    """A malformed line should be ignored, never half-interpreted into a key."""
    assert parse("this is not a setting\nALPHA=one") == {"ALPHA": "one"}


def test_a_value_may_contain_equals_signs() -> None:
    """Keys and connection strings routinely do. Splitting on every `=` would corrupt them."""
    assert parse("URL=postgres://u:p@h/db?a=1&b=2") == {"URL": "postgres://u:p@h/db?a=1&b=2"}


def test_an_empty_value_is_kept_as_empty() -> None:
    assert parse("ALPHA=") == {"ALPHA": ""}


# -- precedence ------------------------------------------------------------


def test_a_real_environment_variable_always_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The most important rule here.

    A config file that beats an explicit `AVATAR_TTS=tone uvicorn ...` is a bad surprise,
    and CI and production set real variables that must not be clobbered by a stray file.
    """
    monkeypatch.setenv("ALPHA", "from-environment")
    env = tmp_path / ".env"
    env.write_text("ALPHA=from-file\n")

    applied = load_env(env)

    assert applied == {}, "nothing should have been applied"
    import os

    assert os.environ["ALPHA"] == "from-environment"


def test_development_overrides_the_base_file(tmp_path: Path) -> None:
    """Layering, not replacement: `.env` holds shared defaults, `.env.development` differs."""
    (tmp_path / ".env").write_text("ALPHA=base\nBETA=shared\n")
    (tmp_path / ".env.development").write_text("ALPHA=development\n")

    applied = {}
    for target in find_env_files(tmp_path / "pkg" / "mod.py"):
        applied.update(load_env(target))

    assert applied["ALPHA"] == "development", "the more specific file must win"
    assert applied["BETA"] == "shared", "and the base file still contributes"


def test_the_candidate_order_is_most_specific_first() -> None:
    assert ENV_FILENAMES.index(".env.development") < ENV_FILENAMES.index(".env")
    assert ENV_FILENAMES.index(".env.local") < ENV_FILENAMES.index(".env")


def test_the_search_stops_at_the_first_directory_with_any_candidate(tmp_path: Path) -> None:
    """
    Prevents a phantom value from a parent directory.

    Without the stop, a `.env` two levels up would layer under the repo's own
    `.env.development` and contribute a setting nobody can account for from inside the
    project.
    """
    (tmp_path / ".env").write_text("ALPHA=from-parent\n")
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / ".env.development").write_text("BETA=from-repo\n")

    found = find_env_files(repo / "src" / "mod.py")

    assert [p.name for p in found] == [".env.development"]
    assert all(p.parent == repo for p in found), "the parent's .env must not be reached"


def test_an_explicit_override_skips_the_search_entirely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller naming a file does not want a merge from a neighbouring one."""
    (tmp_path / ".env").write_text("ALPHA=neighbour\n")
    named = tmp_path / "mounted-secret"
    named.write_text("ALPHA=explicit\n")
    monkeypatch.setenv(ENV_FILE_OVERRIDE, str(named))

    applied = load_env()

    assert applied == {"ALPHA": "explicit"}
    assert loaded_files() == [str(named)]


# -- survivability ---------------------------------------------------------


def test_a_missing_file_is_not_an_error(tmp_path: Path) -> None:
    """A clean clone has no env file and must still run — every default works without one."""
    assert load_env(tmp_path / "does-not-exist") == {}
    assert find_env_files(tmp_path / "nowhere" / "mod.py") == []


def test_an_unreadable_file_is_survived_not_raised(tmp_path: Path) -> None:
    """A permissions problem should degrade to defaults, not refuse to start the server."""
    blocked = tmp_path / ".env"
    blocked.write_text("ALPHA=one\n")
    blocked.chmod(0o000)
    try:
        assert load_env(blocked) == {}
    finally:
        blocked.chmod(0o600)


def test_loaded_files_reports_names_only(tmp_path: Path) -> None:
    """
    `GET /config` surfaces this, so it must never carry contents.

    Most of what these files hold is a credential.
    """
    (tmp_path / ".env.development").write_text("SECRET_KEY=hunter2\n")

    reported = loaded_files(tmp_path / "pkg" / "mod.py")

    assert reported == [str(tmp_path / ".env.development")]
    assert not any("hunter2" in entry for entry in reported)
