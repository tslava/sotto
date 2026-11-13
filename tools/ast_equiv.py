"""AST-equivalence check for the English translation pass (plan 3.3).

    uv run python -m tools.ast_equiv OLD_REF NEW_REF [--paths src tests tools] [--repo DIR]
    uv run python -m tools.ast_equiv --self-test

Compares every ``*.py`` file under ``--paths`` between two git revisions (``WORKTREE`` as a ref
reads the files from disk) and fails unless the trees are identical after this normalisation:

1. Docstrings (the first ``Expr(Constant(str))`` of a module, class or function) are dropped.
   Comments are not in the AST at all.
2. A string at a "message position" that contains Cyrillic in the OLD tree is replaced by a marker
   in both trees. Message positions: the first argument of ``logger.*`` / ``logging.*`` calls,
   the arguments of ``raise X(...)``, the ``help=`` / ``description=`` / ``epilog=`` / ``usage=``
   keywords, ``warnings.warn``, ``sys.exit``, ``pytest.fail`` / ``pytest.skip`` and the message of
   an ``assert``; through ``%``, ``+`` and ``.format`` on a string. In an f-string only the
   constant parts are replaced and the ``{expressions}`` are kept in order.
3. The marker carries the placeholders of the string (``%s``, ``%(name)s``, ``%.2f`` ...; for
   ``help=`` / ``.format`` also ``{}``, ``{0}``, ``{name:spec}``), so the translation has to keep
   the same placeholders. An unknown ``%`` directive fails the check instead of being ignored.
4. Any other Cyrillic string constant left in the OLD tree (outside the exempt paths) is a failure
   of the pre-condition: the tool lists it.

Strict on purpose: if a translation needs a structural change (reordered f-string expressions, a
different call), it is not a translation and belongs in an earlier commit.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import re
import string
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

CYRILLIC = re.compile("[\u0400-\u04ff]")
WORKTREE = "WORKTREE"

# Cyrillic constants are legitimate here (they are compared like any other file).
EXEMPT_PATHS: tuple[str, ...] = (
    "tools/ast_equiv.py",  # its self-test holds Russian sample code
    "tests/test_ast_equiv.py",
    "src/sotto/lang/ru.py",
    "tests/lang/*",
    "tests/legacy/*",
    "tests/fixtures/*",
)

_LOG_RECEIVERS = frozenset({"logger", "logging", "log", "LOGGER", "LOG", "_logger", "_log"})
_LOG_METHODS = frozenset({"debug", "info", "warning", "warn", "error", "exception", "critical"})
_MESSAGE_KEYWORDS = frozenset({"help", "description", "epilog", "usage"})
_BRACE_KEYWORDS = frozenset({"help", "description", "epilog", "usage"})
_PCT_RE = re.compile(
    r"%(?:\(\w+\))?[#0\- +]*(?:\d+|\*)?(?:\.(?:\d+|\*))?[hlL]?([diouxXeEfFgGcrsa%])"
)

# Built in two parts so that this file does not contain the marker as one literal.
_ERROR_MARKER = "<MSG " + "ERROR"

Path_ = tuple[tuple[str, int], ...]  # (field, index) steps from the module; index -1: not a list


class _SignatureError(ValueError):
    pass


def has_cyrillic(text: str) -> bool:
    return CYRILLIC.search(text) is not None


# --------------------------------------------------------------------------- #
# Placeholder signatures
# --------------------------------------------------------------------------- #


def _pct_signature(text: str) -> list[str]:
    found: list[str] = []
    position = 0
    while True:
        index = text.find("%", position)
        if index < 0:
            return sorted(found)
        match = _PCT_RE.match(text, index)
        if match is None:
            raise _SignatureError(f"unknown % directive near {text[index : index + 8]!r}")
        if match.group(1) != "%":  # "%%" is a literal percent sign
            found.append(match.group(0))
        position = match.end()


def _brace_signature(text: str) -> list[str]:
    found: list[str] = []
    try:
        for _, name, spec, conversion in string.Formatter().parse(text):
            if name is not None:
                found.append(f"{{{name}!{conversion or ''}:{spec or ''}}}")
    except ValueError as exc:
        raise _SignatureError(f"bad brace placeholder: {exc}") from exc
    return sorted(found)


def _signature(text: str, *, brace: bool) -> str:
    try:
        parts = _pct_signature(text) + (_brace_signature(text) if brace else [])
    except _SignatureError as exc:
        return f"ERROR {exc}"
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Message positions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Leaf:
    node: ast.AST  # a Constant(str) or a JoinedStr
    brace: bool


def _string_leaves(expr: ast.AST, brace: bool) -> Iterator[_Leaf]:
    """The string pieces of a message expression: constants, f-strings, ``%``/``+`` operands and
    the string a ``.format`` is called on."""
    if (isinstance(expr, ast.Constant) and isinstance(expr.value, str)) or isinstance(
        expr, ast.JoinedStr
    ):
        yield _Leaf(expr, brace)
    elif isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Mod | ast.Add):
        yield from _string_leaves(expr.left, brace)
        if isinstance(expr.op, ast.Add):
            yield from _string_leaves(expr.right, brace)
    elif isinstance(expr, ast.IfExp):
        yield from _string_leaves(expr.body, brace)
        yield from _string_leaves(expr.orelse, brace)
    elif (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Attribute)
        and expr.func.attr == "format"
    ):
        yield from _string_leaves(expr.func.value, True)


def _call_positions(call: ast.Call) -> Iterator[_Leaf]:
    func = call.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        receiver, method = func.value.id, func.attr
        if receiver in _LOG_RECEIVERS and method in _LOG_METHODS and call.args:
            yield from _string_leaves(call.args[0], False)
        elif receiver in _LOG_RECEIVERS and method == "log" and len(call.args) > 1:
            yield from _string_leaves(call.args[1], False)
        elif (receiver, method) in {("warnings", "warn"), ("sys", "exit")} and call.args:
            yield from _string_leaves(call.args[0], False)
        elif receiver == "pytest" and method in {"fail", "skip", "exit"}:
            if call.args:
                yield from _string_leaves(call.args[0], False)
            for keyword in call.keywords:
                if keyword.arg in {"reason", "msg"}:
                    yield from _string_leaves(keyword.value, False)
    for keyword in call.keywords:
        if keyword.arg in _MESSAGE_KEYWORDS:
            yield from _string_leaves(keyword.value, keyword.arg in _BRACE_KEYWORDS)


def message_leaves(tree: ast.AST) -> list[_Leaf]:
    leaves: list[_Leaf] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            leaves.extend(_call_positions(node))
        elif isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            for arg in node.exc.args:
                leaves.extend(_string_leaves(arg, False))
            for keyword in node.exc.keywords:
                leaves.extend(_string_leaves(keyword.value, False))
        elif isinstance(node, ast.Assert) and node.msg is not None:
            leaves.extend(_string_leaves(node.msg, False))
    return leaves


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


def _paths(tree: ast.AST) -> dict[int, Path_]:
    """``id(node) -> path`` for every node, so a node of the OLD tree can be found in the NEW."""
    result: dict[int, Path_] = {}

    def visit(node: ast.AST, path: Path_) -> None:
        result[id(node)] = path
        for name, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                visit(value, (*path, (name, -1)))
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, ast.AST):
                        visit(item, (*path, (name, index)))

    visit(tree, ())
    return result


def _drop_docstrings(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body[0] = ast.copy_location(ast.Pass(), body[0])


def _marker(node: ast.AST, brace: bool) -> ast.AST:
    """The replacement of a message string: its placeholders only."""
    if isinstance(node, ast.Constant):
        text = str(node.value)
        return ast.copy_location(ast.Constant(f"<MSG {_signature(text, brace=brace)}>"), node)
    assert isinstance(node, ast.JoinedStr)
    text = "".join(
        str(part.value) if isinstance(part, ast.Constant) else "\0"
        for part in node.values
        if isinstance(part, ast.Constant | ast.FormattedValue)
    )
    kept = [part for part in node.values if isinstance(part, ast.FormattedValue)]
    head = ast.Constant(f"<MSG {_signature(text, brace=brace)}>")
    return ast.copy_location(ast.JoinedStr([head, *kept]), node)


def _replace(tree: ast.AST, targets: dict[Path_, bool]) -> None:
    """Replace the nodes at the given paths (path -> brace flag) by their markers, in place."""

    def visit(node: ast.AST, path: Path_) -> ast.AST:
        if (
            path in targets
            and isinstance(node, ast.Constant | ast.JoinedStr)
            and not (isinstance(node, ast.Constant) and not isinstance(node.value, str))
        ):
            return _marker(node, targets[path])
        for name, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                setattr(node, name, visit(value, (*path, (name, -1))))
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, ast.AST):
                        value[index] = visit(item, (*path, (name, index)))
        return node

    visit(tree, ())


def _cyrillic_leaves(leaf: _Leaf) -> bool:
    node = leaf.node
    if isinstance(node, ast.Constant):
        return has_cyrillic(str(node.value))
    assert isinstance(node, ast.JoinedStr)
    return any(
        isinstance(part, ast.Constant) and has_cyrillic(str(part.value)) for part in node.values
    )


def _leftover_cyrillic(tree: ast.AST, source: str) -> list[int]:
    """Line numbers of Cyrillic string constants left after the normalisation. Only strings
    TYPED in Cyrillic count: ``"\\u0442"`` escapes are code that handles Cyrillic, not text to
    translate (the tools and tests of the language gate are written that way)."""
    return sorted(
        {
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and has_cyrillic(node.value)
            and has_cyrillic(ast.get_source_segment(source, node) or "")
        }
    )


def _first_diff(old: object, new: object, trail: str = "", lines: str = "") -> str | None:
    """Describe the first place where two normalised trees differ (and where, by line)."""
    if type(old) is not type(new):
        return f"{trail}: {type(old).__name__} (old) vs {type(new).__name__} (new) [{lines}]"
    if isinstance(old, ast.AST) and isinstance(new, ast.AST):
        if hasattr(old, "lineno"):
            lines = f"old line {old.lineno}, new line {getattr(new, 'lineno', '?')}"
        for name, old_value in ast.iter_fields(old):
            found = _first_diff(old_value, getattr(new, name, None), f"{trail}/{name}", lines)
            if found is not None:
                return found
        return None
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return f"{trail}: {len(old)} items (old) vs {len(new)} (new) [{lines}]"
        for index, (a, b) in enumerate(zip(old, new, strict=True)):
            found = _first_diff(a, b, f"{trail}[{index}]", lines)
            if found is not None:
                return found
        return None
    return None if old == new else f"{trail}: {old!r} (old) vs {new!r} (new) [{lines}]"


def compare_sources(old: str, new: str, *, exempt: bool = False) -> list[str]:
    """Problems that make ``new`` more than a translation of ``old`` (empty: equivalent)."""
    try:
        old_tree = ast.parse(old)
        new_tree = ast.parse(new)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    problems: list[str] = []
    _drop_docstrings(old_tree)
    _drop_docstrings(new_tree)
    old_paths = _paths(old_tree)
    targets = {
        old_paths[id(leaf.node)]: leaf.brace
        for leaf in message_leaves(old_tree)
        if _cyrillic_leaves(leaf)
    }
    _replace(old_tree, targets)
    _replace(new_tree, targets)
    if not exempt:
        problems += [
            f"line {line}: Cyrillic string outside a message position (pre-condition)"
            for line in _leftover_cyrillic(old_tree, old)
        ]
    for tree, label in ((old_tree, "old"), (new_tree, "new")):
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith(_ERROR_MARKER)
            ):
                problems.append(f"{label} line {node.lineno}: {node.value[5:-1]}")
    if ast.dump(old_tree) != ast.dump(new_tree):
        problems.append(f"trees differ: {_first_diff(old_tree, new_tree)}")
    return problems


# --------------------------------------------------------------------------- #
# Git revisions
# --------------------------------------------------------------------------- #


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout


def list_python_files(repo: Path, ref: str, paths: Sequence[str]) -> list[str]:
    if ref == WORKTREE:
        found: list[str] = []
        for base in paths:
            root = repo / base
            candidates = [root] if root.is_file() else sorted(root.rglob("*.py"))
            found += [
                p.relative_to(repo).as_posix() for p in candidates if "__pycache__" not in p.parts
            ]
        return sorted(set(found))
    out = _git(repo, "ls-tree", "-r", "--name-only", "-z", ref, "--", *paths)
    return sorted(n for n in out.decode().split("\0") if n.endswith(".py"))


def read_file(repo: Path, ref: str, name: str) -> str:
    if ref == WORKTREE:
        return (repo / name).read_text(encoding="utf-8")
    return _git(repo, "show", f"{ref}:{name}").decode("utf-8")


def compare_revisions(
    repo: Path, old_ref: str, new_ref: str, paths: Sequence[str]
) -> dict[str, list[str]]:
    """File name -> problems, for the files that are not equivalent."""
    old_files = set(list_python_files(repo, old_ref, paths))
    new_files = set(list_python_files(repo, new_ref, paths))
    result: dict[str, list[str]] = {}
    for name in sorted(old_files - new_files):
        result[name] = ["file removed"]
    for name in sorted(new_files - old_files):
        result[name] = ["file added"]
    for name in sorted(old_files & new_files):
        exempt = any(fnmatch.fnmatch(name, pattern) for pattern in EXEMPT_PATHS)
        problems = compare_sources(
            read_file(repo, old_ref, name), read_file(repo, new_ref, name), exempt=exempt
        )
        if problems:
            result[name] = problems
    return result


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

_OLD = '''\
"""Модуль."""
import logging
logger = logging.getLogger(__name__)

def f(x):
    """Док."""
    logger.info("получено %s из %d", x, 3)
    if x:
        raise ValueError(f"плохо {x!r}: {x + 1}")
    return x
'''

_GOOD = '''\
"""Module."""
import logging
logger = logging.getLogger(__name__)

def f(x):
    """Doc."""
    logger.info("got %s of %d", x, 3)
    if x:
        raise ValueError(f"bad {x!r}: {x + 1}")
    return x
'''


def self_test() -> list[str]:
    """Check the checker: cases that must pass and cases that must fail. Returns failures."""
    must_pass = {
        "translation": (_OLD, _GOOD),
        "identical": (_GOOD, _GOOD),
        "comment only": (_GOOD, _GOOD.replace("def f", "# note\ndef f")),
        "argparse help": (
            'p.add_argument("--x", help="путь %(default)s и {0}")\n',
            'p.add_argument("--x", help="path %(default)s and {0}")\n',
        ),
    }
    must_fail = {
        "code changed": (_OLD, _GOOD.replace("return x", "return x + 1")),
        "placeholder lost": (_OLD, _GOOD.replace("%s of %d", "%s of")),
        "placeholder added": (_OLD, _GOOD.replace("%s of %d", "%s of %d %s")),
        "fstring expression changed": (_OLD, _GOOD.replace("{x + 1}", "{x + 2}")),
        "fstring expressions reordered": (
            'raise ValueError(f"а {x} б {y}")\n',
            'raise ValueError(f"b {y} a {x}")\n',
        ),
        "unknown directive": (
            'logger.info("проценты 50% готово")\n',
            'logger.info("percent 50% done")\n',
        ),
        "cyrillic outside a message": ('x = "привет"\n', 'x = "hello"\n'),
        "string changed outside a message": ('x = "hello"\n', 'x = "hallo"\n'),
        "message in english stays checked": (_GOOD, _GOOD.replace('"got %s', '"got %r')),
        "brace placeholder changed": (
            'p.add_argument("--x", help="путь {0}")\n',
            'p.add_argument("--x", help="path {1}")\n',
        ),
        "syntax error": (_GOOD, "def f(:\n"),
    }
    failures: list[str] = []
    for name, (old, new) in must_pass.items():
        problems = compare_sources(old, new)
        if problems:
            failures.append(f"{name}: expected a pass, got {problems}")
    for name, (old, new) in must_fail.items():
        if not compare_sources(old, new):
            failures.append(f"{name}: expected a failure, got a pass")
    with tempfile.TemporaryDirectory() as raw:
        repo = Path(raw)
        _git(repo, "init", "-q")
        (repo / "m.py").write_text(_OLD, encoding="utf-8")
        _git(repo, "add", ".")
        commit = ["-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm"]
        _git(repo, *commit, "old")
        (repo / "m.py").write_text(_GOOD, encoding="utf-8")
        _git(repo, "add", ".")
        _git(repo, *commit, "new")
        if compare_revisions(repo, "HEAD~1", "HEAD", ["."]):
            failures.append("git revisions: a translation was reported")
        (repo / "m.py").write_text(_GOOD.replace("return x", "return None"), encoding="utf-8")
        if "m.py" not in compare_revisions(repo, "HEAD~1", WORKTREE, ["."]):
            failures.append("git revisions: a code change in the worktree went unnoticed")
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.ast_equiv", description=__doc__)
    parser.add_argument("old_ref", nargs="?")
    parser.add_argument("new_ref", nargs="?")
    parser.add_argument("--paths", nargs="+", default=["src", "tests", "tools"])
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--self-test", action="store_true", help="check the checker and exit")
    args = parser.parse_args(argv)
    if args.self_test:
        failures = self_test()
        for failure in failures:
            print(f"self-test FAILED: {failure}")
        if not failures:
            print("ast_equiv self-test OK")
        return 1 if failures else 0
    if not args.old_ref or not args.new_ref:
        parser.error("OLD_REF and NEW_REF are required (or --self-test)")
    result = compare_revisions(args.repo, args.old_ref, args.new_ref, args.paths)
    for name, problems in result.items():
        print(name)
        for problem in problems:
            print(f"  {problem}")
    total = len(list_python_files(args.repo, args.old_ref, args.paths))
    if result:
        print(f"ast_equiv FAILED: {len(result)} of {total} files differ")
        return 1
    print(f"ast_equiv OK: {total} files equivalent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
