"""tools/ast_equiv.py: the AST-equivalence check of the English translation pass."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from tools import ast_equiv
from tools.ast_equiv import WORKTREE, compare_revisions, compare_sources, main, self_test

OLD = '''\
"""Модуль."""
import logging
logger = logging.getLogger(__name__)


def f(x):
    """Док."""
    logger.info("получено %s из %d", x, 3)  # комментарий
    if x:
        raise ValueError(f"плохо {x!r}: {x + 1}")
    return x
'''

NEW = '''\
"""Module."""
import logging
logger = logging.getLogger(__name__)


def f(x):
    """Doc."""
    logger.info("got %s of %d", x, 3)  # comment
    if x:
        raise ValueError(f"bad {x!r}: {x + 1}")
    return x
'''


def test_self_test_passes() -> None:
    assert self_test() == []
    assert main(["--self-test"]) == 0


def test_translation_of_docstrings_comments_and_messages_is_equivalent() -> None:
    assert compare_sources(OLD, NEW) == []


@pytest.mark.parametrize(
    ("new", "needle"),
    [
        (NEW.replace("return x", "return x + 1"), "trees differ"),
        (NEW.replace("%s of %d", "%s of"), "trees differ"),
        (NEW.replace("{x + 1}", "{x - 1}"), "trees differ"),
        (NEW.replace("{x!r}", "{x!s}"), "trees differ"),
        (NEW.replace("logger.info", "logger.debug"), "trees differ"),
        ("def f(:\n", "syntax error"),
    ],
)
def test_real_changes_are_reported(new: str, needle: str) -> None:
    problems = compare_sources(OLD, new)
    assert problems and needle in problems[0]


def test_the_report_names_the_first_difference_with_line_numbers() -> None:
    (problem,) = compare_sources(OLD, NEW.replace("return x", "return x + 1"))
    assert "old line" in problem and "new line" in problem


def test_unknown_percent_directive_fails_instead_of_passing_silently() -> None:
    problems = compare_sources('logger.info("50% готово")\n', 'logger.info("50% done")\n')
    assert any("unknown % directive" in problem for problem in problems)


def test_brace_placeholders_are_checked_in_help_and_format() -> None:
    old = 'p.add_argument("--x", help="путь {0}")\n'
    assert compare_sources(old, 'p.add_argument("--x", help="path {0}")\n') == []
    assert compare_sources(old, 'p.add_argument("--x", help="path {}")\n') != []
    old = 'raise ValueError("ошибка {name:>5}".format(name=n))\n'
    assert compare_sources(old, 'raise ValueError("error {name:>5}".format(name=n))\n') == []
    assert compare_sources(old, 'raise ValueError("error {name}".format(name=n))\n') != []


def test_message_positions_cover_warn_exit_assert_and_percent_operator() -> None:
    cases = [
        ('warnings.warn("внимание")\n', 'warnings.warn("attention")\n'),
        ('sys.exit("конец")\n', 'sys.exit("end")\n'),
        ('assert x, "нет"\n', 'assert x, "no"\n'),
        ('raise SystemExit("ключ %s" % key)\n', 'raise SystemExit("key %s" % key)\n'),
        ('logger.log(20, "сообщение")\n', 'logger.log(20, "message")\n'),
        ('pytest.fail("сбой")\n', 'pytest.fail("failure")\n'),
    ]
    for old, new in cases:
        assert compare_sources(old, new) == [], old


def test_cyrillic_outside_a_message_is_a_precondition_failure_unless_exempt() -> None:
    old, new = 'x = "привет"\n', 'x = "hello"\n'
    assert any("pre-condition" in p for p in compare_sources(old, new))
    # An exempt file is still compared, only the pre-condition is waived.
    assert all("pre-condition" not in p for p in compare_sources(old, new, exempt=True))
    assert compare_sources(old, old, exempt=True) == []


def test_cyrillic_written_as_escapes_is_not_text_to_translate() -> None:
    source = 'x = "\\u043f\\u0440\\u0438\\u0432\\u0435\\u0442"\n'
    assert compare_sources(source, source) == []


def _git(repo: Path, *args: str) -> None:
    config = ["-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run(["git", "-C", str(repo), *config, *args], check=True, capture_output=True)


def test_revisions_files_added_removed_and_worktree(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.py").write_text(OLD, encoding="utf-8")
    (tmp_path / "gone.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "old")
    (tmp_path / "a.py").write_text(NEW, encoding="utf-8")
    (tmp_path / "gone.py").unlink()
    (tmp_path / "new.py").write_text("y = 2\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "new")

    result = compare_revisions(tmp_path, "HEAD~1", "HEAD", ["."])
    assert result == {"gone.py": ["file removed"], "new.py": ["file added"]}

    (tmp_path / "gone.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "new.py").unlink()
    assert compare_revisions(tmp_path, "HEAD~1", WORKTREE, ["."]) == {}
    (tmp_path / "a.py").write_text(NEW.replace("return x", "return 0"), encoding="utf-8")
    assert list(compare_revisions(tmp_path, "HEAD~1", WORKTREE, ["."])) == ["a.py"]


def test_main_reports_and_sets_the_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.py").write_text(OLD, encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "old")
    (tmp_path / "a.py").write_text(NEW, encoding="utf-8")
    _git(tmp_path, "commit", "-qam", "new")
    assert main(["HEAD~1", "HEAD", "--repo", str(tmp_path), "--paths", "."]) == 0
    assert "ast_equiv OK: 1 files" in capsys.readouterr().out
    (tmp_path / "a.py").write_text(NEW.replace("return x", "return 0"), encoding="utf-8")
    assert main(["HEAD~1", WORKTREE, "--repo", str(tmp_path), "--paths", "."]) == 1
    out = capsys.readouterr().out
    assert "a.py" in out and "ast_equiv FAILED" in out
    with pytest.raises(SystemExit):
        main([])


def test_exempt_paths_match_the_language_gate_allowance() -> None:
    from tools import check_language

    for pattern in ast_equiv.EXEMPT_PATHS:
        if pattern.startswith(("tests/lang", "tests/fixtures")):
            assert check_language.is_allowed(pattern.replace("*", "x.py"))
