"""Owner-UI catalog helper: draft a catalog from source literals, check a catalog against them.

The owner-facing wording of the engine used to be string literals in ``commands.py``, ``alerts.py``
and ``stores.py``. PR-5a moved it into ``locales/en`` (English) and the generated
``locale/ru`` of the legacy pack (the former literals, byte for byte, as a migration proof).
This tool reads sources with ``ast`` (nothing is imported or executed):

* ``draft``: print a YAML draft of the Cyrillic literals of a source, in source order, f-string
  expressions turned into ``{name}`` placeholders (a starting point for hand-curated keys);
* ``check``: verify a catalog directory against sources, in both directions. Every Cyrillic chunk of
  a source literal (the text between f-string expressions, whitespace stripped) must appear inside
  some catalog value (``{...}`` normalised to ``{}``), and every catalog value with Cyrillic must
  contain at least one source literal (nothing invented). Run it against the sources as they were
  BEFORE the refactor (``git show <ref>:src/sotto/commands.py``). Values are never printed, only
  keys and line numbers.

Usage::

    python -m tools.extract_locale draft SOURCE.py
    python -m tools.extract_locale check --catalog personas/_legacy_ru/locale/ru SOURCE.py [...]
"""

from __future__ import annotations

import argparse
import ast
import re
import string
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from sotto.i18n import read_locale_dir

_CYRILLIC = re.compile("[А-Яа-яЁё]")
_LOG_METHODS = frozenset({"debug", "info", "warning", "error", "exception", "critical", "log"})
_PLACEHOLDER = "{}"


@dataclass(frozen=True)
class SourceLiteral:
    """A string literal (or f-string) of a source file; expressions are ``{}``."""

    function: str
    lineno: int
    template: str
    names: tuple[str, ...]


def _expr_name(node: ast.expr, index: int) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _expr_name(node.func, index)
    return f"expr{index}"


def _joined(node: ast.JoinedStr) -> tuple[str, tuple[str, ...]]:
    text: list[str] = []
    names: list[str] = []
    for part in node.values:
        if isinstance(part, ast.Constant):
            text.append(str(part.value))
        elif isinstance(part, ast.FormattedValue):
            text.append(_PLACEHOLDER)
            names.append(_expr_name(part.value, len(names) + 1))
    return "".join(text), tuple(names)


def _is_logging_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _LOG_METHODS
    )


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def extract_literals(source: str) -> list[SourceLiteral]:
    """Cyrillic-bearing string literals outside docstrings and logger calls, in source order."""
    tree = ast.parse(source)
    skip = _docstring_nodes(tree)
    found: list[SourceLiteral] = []

    def visit(node: ast.AST, function: str) -> None:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            function = node.name
        if id(node) in skip or _is_logging_call(node):
            return
        if isinstance(node, ast.JoinedStr):
            template, names = _joined(node)
            if _CYRILLIC.search(template):
                found.append(SourceLiteral(function, node.lineno, template, names))
            # Literals inside the expressions ({'да' if x else 'нет'}) are literals too.
            for part in node.values:
                if isinstance(part, ast.FormattedValue):
                    visit(part.value, function)
            return
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _CYRILLIC.search(node.value):
                found.append(SourceLiteral(function, node.lineno, node.value, ()))
            return
        for child in ast.iter_child_nodes(node):
            visit(child, function)

    visit(tree, "<module>")
    return sorted(found, key=lambda lit: lit.lineno)


def normalize_value(value: str) -> str:
    """A catalog value with every ``{field}`` replaced by ``{}`` and ``{{``/``}}`` unescaped."""
    out: list[str] = []
    for literal, field, _spec, _conv in string.Formatter().parse(value):
        out.append(literal)
        if field is not None:
            out.append(_PLACEHOLDER)
    return "".join(out)


def _catalog_values(directory: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for key, value in read_locale_dir(directory).items():
        forms = [value] if isinstance(value, str) else list(value.values())
        for index, form in enumerate(forms):
            values[f"{key}#{index}" if len(forms) > 1 else key] = normalize_value(form)
    return values


def _pieces(literal: SourceLiteral) -> list[str]:
    """The checkable chunks of a literal: the text between placeholders, stripped, with Cyrillic."""
    chunks = (chunk.strip() for chunk in literal.template.split(_PLACEHOLDER))
    return [chunk for chunk in chunks if _CYRILLIC.search(chunk)]


def check_catalog(directory: Path, sources: Sequence[tuple[str, str]]) -> list[str]:
    """Problems found comparing ``directory`` with ``(name, source_text)`` pairs; empty = fine."""
    values = _catalog_values(directory)
    normalized = list(values.values())
    literals = [(name, lit) for name, text in sources for lit in extract_literals(text)]
    errors: list[str] = []
    for name, literal in literals:
        for piece in _pieces(literal):
            if not any(piece in value for value in normalized):
                errors.append(
                    f"{name}:{literal.lineno} ({literal.function}): literal not in the catalog"
                )
    for key, value in values.items():
        if not _CYRILLIC.search(value):
            continue
        if not any(piece in value for _, lit in literals for piece in _pieces(lit)):
            errors.append(f"{key}: value has no source literal inside it")
    return errors


def draft_catalog(source: str, name: str) -> dict[str, dict[str, str]]:
    """``{file: {function_N: template}}`` with ``{name}`` placeholders inferred from expressions."""
    entries: dict[str, str] = {}
    counts: dict[str, int] = {}
    for literal in extract_literals(source):
        template = literal.template
        for placeholder_name in literal.names:
            template = template.replace(_PLACEHOLDER, "{" + placeholder_name + "}", 1)
        counts[literal.function] = counts.get(literal.function, 0) + 1
        entries[f"{literal.function.lstrip('_')}_{counts[literal.function]}"] = template
    return {name: entries}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _iter_sources(paths: Sequence[Path]) -> Iterator[tuple[str, str]]:
    for path in paths:
        yield path.name, _read(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.extract_locale", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    draft = sub.add_parser("draft", help="print a YAML draft of the Cyrillic literals")
    draft.add_argument("source", type=Path)
    check = sub.add_parser("check", help="verify a catalog directory against sources")
    check.add_argument("--catalog", type=Path, required=True, help="locale directory (*.yaml)")
    check.add_argument("sources", type=Path, nargs="+")
    args = parser.parse_args(argv)

    if args.command == "draft":
        print(
            yaml.safe_dump(
                draft_catalog(_read(args.source), args.source.stem),
                allow_unicode=True,
                sort_keys=False,
                width=100000,
            ),
            end="",
        )
        return 0
    errors = check_catalog(args.catalog, list(_iter_sources(args.sources)))
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    print(f"catalog {args.catalog} matches {len(args.sources)} source file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
