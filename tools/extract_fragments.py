"""One-shot extraction of ``fragments.yaml`` from engine sources (plan 04, PR-3).

Before PR-3 the wording that goes into model prompts or into the database lived as string
constants and literals in ``src/sotto/*.py``. This tool reads them with ``ast`` (nothing is
imported or executed) and writes the pack's ``fragments.yaml`` VERBATIM: the real pack and the
synthetic pack are both generated from the same pre-PR-3 sources, so the golden compare stays
empty. After PR-3 the engine reads the pack, the constants are gone, and the tool only runs
against a source tree that still has them (``--src PATH`` points at such a tree).

Usage::

    python -m tools.extract_fragments --src SRC_DIR --out fragments.yaml [--package NAME]

``SRC_DIR`` is the directory holding the package (``src``); ``--package`` is its name
(``sotto`` by default; the pre-rewrite project used another name). Values are never printed.
Several sources per key are cross-checked; a mismatch is reported and fails the run (exit 2).
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_CYRILLIC = re.compile("[\u0410-\u044f\u0401\u0451]")


class ExtractError(Exception):
    """A source the spec points at is missing or has an unexpected shape."""


@dataclass(frozen=True)
class Const:
    """Module-level assignment ``NAME = <literal expression>``."""

    module: str
    name: str


@dataclass(frozen=True)
class Inline:
    """The ``index``-th string literal in a function body that matches ``pattern``.

    Literals are ordered by position (line, column); docstrings are skipped. Parts of an
    f-string count as literals, which is how ``f"now {x}"`` is read.
    """

    module: str
    function: str
    index: int
    pattern: str = "[\u0410-\u044f\u0401\u0451]"


@dataclass(frozen=True)
class FString:
    """The first f-string with Cyrillic text in a function; each ``{expr}`` becomes
    ``{params[i]}``. ``strip_suffix`` is cut off the end of the template if given."""

    module: str
    function: str
    params: tuple[str, ...]
    strip_suffix: str = ""


Source = Const | Inline | FString


@dataclass(frozen=True)
class Item:
    key: str
    source: Source
    cross: tuple[Source, ...] = ()
    # "equal": cross sources must equal the value; "prefix": value starts with them
    # (up to the first placeholder).
    mode: str = "equal"


def _c(module: str, name: str) -> Const:
    return Const(module, name)


ITEMS: tuple[Item, ...] = (
    # placeholders (persisted into messages.text / bot_replies.text / display_name)
    Item(
        "placeholders.photo",
        Inline("sanitize", "media_placeholder", 1),
        (_c("vision", "_PHOTO_PLACEHOLDER"), Inline("export_parser", "_media_placeholder", 0)),
    ),
    Item(
        "placeholders.sticker",
        Inline("sanitize", "media_placeholder", 0),
        (Inline("export_parser", "_media_placeholder", 1),),
    ),
    Item(
        "placeholders.voice",
        Inline("sanitize", "media_placeholder", 2),
        (
            Inline("export_parser", "_media_placeholder", 2),
            Inline("export_parser", "_media_placeholder", 4),
        ),
    ),
    Item(
        "placeholders.video",
        Inline("sanitize", "media_placeholder", 3),
        (Inline("export_parser", "_media_placeholder", 3),),
    ),
    Item(
        "placeholders.file",
        Inline("sanitize", "media_placeholder", 4),
        (Inline("export_parser", "_media_placeholder", 5),),
    ),
    Item("placeholders.photo_with_desc", FString("vision", "photo_text", ("desc",))),
    Item("placeholders.no_caption", _c("vision", "_CAPTION_EMPTY")),
    Item("placeholders.participant", FString("sanitize", "sanitize_display_name", ("n",))),
    Item(
        "placeholders.sticker_tag",
        FString("responder", "_generate_and_send", ("id", "text"), strip_suffix=" {text}"),
        (
            _c("postprocess", "_STICKER_PLACEHOLDER_PREFIX"),
            _c("motifs", "_STICKER_PLACEHOLDER_PREFIX"),
            _c("vocab", "_STICKER_PREFIX"),
        ),
        mode="prefix",
    ),
    Item("placeholders.recent_replies_empty", _c("prompt", "_RECENT_REPLIES_EMPTY")),
    Item(
        "placeholders.context_empty",
        _c("followup", "_CONTEXT_EMPTY"),
        (_c("reactions", "_CONTEXT_EMPTY"),),
    ),
    Item(
        "placeholders.messages_empty",
        _c("followup", "_MESSAGES_EMPTY"),
    ),
    Item("placeholders.reply_to_none", _c("reactions", "_REPLY_TO_NONE")),
    Item("placeholders.reply_to_unavailable", _c("reactions", "_REPLY_TO_UNAVAILABLE")),
    # /ex add fallback pair; read from the pre-PR-5a commands.py (the owner catalog took them over
    # for one release and PR-5b moved them back to the pack: they end up in few-shot data).
    Item("placeholders.ex_add_name", Inline("commands", "_cmd_ex_add", 0)),
    Item("placeholders.ex_add_user", Inline("commands", "_cmd_ex_add", 1)),
    # situations
    Item("situations.ambient", _c("prompt", "SITUATION_AMBIENT")),
    Item("situations.late", _c("prompt", "SITUATION_LATE")),
    Item("situations.morning", _c("prompt", "SITUATION_MORNING")),
    Item("situations.spontaneous", _c("prompt", "SITUATION_SPONTANEOUS")),
    Item("situations.life", _c("prompt", "SITUATION_LIFE_TEMPLATE")),
    Item("situations.addressed_single", _c("prompt", "SITUATION_ADDRESSED_SINGLE")),
    Item("situations.addressed_multi_header", _c("prompt", "SITUATION_ADDRESSED_MULTI_HEADER")),
    Item("situations.addressed_multi_footer", _c("prompt", "SITUATION_ADDRESSED_MULTI_FOOTER")),
    Item("situations.followup_single", _c("prompt", "SITUATION_FOLLOWUP_SINGLE")),
    Item("situations.followup_multi_header", _c("prompt", "SITUATION_FOLLOWUP_MULTI_HEADER")),
    Item("situations.followup_multi_footer", _c("prompt", "SITUATION_FOLLOWUP_MULTI_FOOTER")),
    Item("situations.checkin_header", _c("prompt", "SITUATION_CHECKIN_HEADER")),
    Item("situations.checkin_footer", _c("prompt", "SITUATION_CHECKIN_FOOTER")),
    Item("situations.callback", _c("prompt", "SITUATION_CALLBACK_TEMPLATE")),
    # prompt scaffolding
    Item("prompt.places_none", _c("prompt", "PLACES_NONE")),
    Item("prompt.data_notice", _c("prompt", "_DATA_DISCLAIMER")),
    Item("prompt.recent_replies_label", _c("prompt", "_RECENT_REPLIES_LABEL")),
    Item("prompt.json_reminder", _c("prompt", "_JSON_REMINDER")),
    Item("prompt.json_reminder_checkin", _c("prompt", "JSON_REMINDER_CHECKIN")),
    Item("prompt.context_marker", _c("prompt", "_CONTEXT_MARKER")),
    Item("prompt.recent_replies_marker", _c("prompt", "_RECENT_REPLIES_MARKER")),
    Item("prompt.places_marker", _c("prompt", "_PLACES_MARKER")),
    Item("prompt.no_target", _c("judge", "_NO_TRIGGER"), (_c("stickers", "_NO_TRIGGER"),)),
    # render headers and pieces
    Item("render.life_header", _c("prompt", "_LIFE_HEADER")),
    Item("render.jokes_header", _c("jokes", "_HEADER")),
    Item("render.chat_memory_header", _c("chat_memory", "_HEADER")),
    Item("render.diary_today_header", _c("diary", "_TODAY_HEADER")),
    Item("render.diary_week_header", _c("diary", "_WEEK_HEADER")),
    Item("render.weekdays", _c("diary", "_WEEKDAYS")),
    Item("render.avoid_template", _c("motifs", "_AVOID_TEMPLATE")),
    Item("render.avoid_no_story", _c("motifs", "_AVOID_NO_STORY")),
    Item("render.accusative", _c("motifs", "_ACCUSATIVE")),
    Item(
        "render.self_share",
        FString(
            "motifs",
            "render_self_share",
            (
                "days",
                "ambient_hit",
                "ambient_total",
                "address_hit",
                "address_total",
                "other_hit",
                "other_total",
            ),
        ),
    ),
    Item("render.places_block_header", Inline("places", "render_places_block", 0)),
    Item("render.places_menu_header", _c("places", "_MENU_HEADER")),
    Item("render.places_quiet", Inline("places", "render_places_menu", 0)),
    Item("render.places_noisy", Inline("places", "render_places_menu", 1)),
    Item("render.quote_open", Inline("jokes", "render_jokes", 0, "«")),
    Item("render.quote_close", Inline("jokes", "render_jokes", 0, "»")),
    # weather pieces
    Item("weather.wmo", _c("weather", "WMO")),
    Item("weather.home_prefix", Inline("weather", "render_weather", 2)),
    Item(
        "weather.now_word",
        Inline("weather", "render_weather", 0),
        (Inline("weather", "render_weather_place", 0),),
    ),
    Item(
        "weather.and_word",
        Inline("weather", "render_weather", 1),
        (Inline("weather", "render_weather_place", 1),),
    ),
    Item("weather.today_from", Inline("weather", "render_weather", 3)),
    Item(
        "weather.range_to",
        Inline("weather", "render_weather", 4),
        (Inline("weather", "render_weather", 6), Inline("weather", "render_weather_place", 3)),
    ),
    Item("weather.tomorrow_from", Inline("weather", "render_weather", 5)),
    Item("weather.tomorrow_from_lower", Inline("weather", "render_weather_place", 2)),
    Item("weather.place_prefix", Inline("weather", "render_weather_place", 4)),
    # vocabulary
    Item("vocab.offer_header", _c("vocab", "_OFFER_HEADER")),
    Item("vocab.offer_tail", _c("vocab", "_OFFER_TAIL")),
    Item("vocab.request_header", _c("vocab", "_REQUEST_HEADER")),
    Item("vocab.request_tail", _c("vocab", "_REQUEST_TAIL")),
    Item("vocab.meanings_header", _c("vocab", "_MEANINGS_HEADER")),
    # offline tools
    Item("tools.places_fact_prompt", _c("places_fill", "_FACT_INSTRUCTION")),
    Item("tools.places_reviews_preamble", _c("places_fill", "_REVIEWS_PREAMBLE")),
    Item("tools.sticker_ocr_prompt", _c("stickers_fill", "_RECOGNITION_INSTRUCTION")),
)

_KEYS = [item.key for item in ITEMS]


@dataclass
class _Module:
    tree: ast.Module
    consts: dict[str, ast.expr] = field(default_factory=dict)


def _load(src: Path, package: str, module: str, cache: dict[str, _Module]) -> _Module:
    if module not in cache:
        path = src / package / f"{module}.py"
        if not path.is_file():
            raise ExtractError(f"source module {module}.py not found under {src / package}")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        consts: dict[str, ast.expr] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    consts[target.id] = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.value is not None:
                    consts[node.target.id] = node.value
        cache[module] = _Module(tree, consts)
    return cache[module]


def _eval(node: ast.expr, mod: _Module) -> Any:
    """Literal evaluation with ``+`` on strings and references to module constants."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _eval(node.left, mod), _eval(node.right, mod)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
        raise ExtractError("unsupported '+' in a constant")
    if isinstance(node, ast.Tuple | ast.List):
        return [_eval(item, mod) for item in node.elts]
    if isinstance(node, ast.Dict):
        out: dict[Any, Any] = {}
        for key, value in zip(node.keys, node.values, strict=True):
            if key is None:
                raise ExtractError("dict unpacking in a constant")
            out[_eval(key, mod)] = _eval(value, mod)
        return out
    if isinstance(node, ast.Name) and node.id in mod.consts:
        return _eval(mod.consts[node.id], mod)
    raise ExtractError(f"unsupported expression {type(node).__name__} in a constant")


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name:
            return node
    raise ExtractError(f"function {name} not found")


def _docstring_ids(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[int]:
    body = func.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        return {id(body[0].value)}
    return set()


def _inline(source: Inline, mod: _Module) -> str:
    func = _find_function(mod.tree, source.function)
    skip = _docstring_ids(func)
    pattern = re.compile(source.pattern)
    found = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in skip
        and pattern.search(node.value)
    ]
    found.sort(key=lambda n: (n.lineno, n.col_offset))
    if source.index >= len(found):
        raise ExtractError(f"{source.function}: literal #{source.index} not found")
    return str(found[source.index].value)


def _fstring(source: FString, mod: _Module) -> str:
    func = _find_function(mod.tree, source.function)
    for node in ast.walk(func):
        if not isinstance(node, ast.JoinedStr):
            continue
        if not any(
            isinstance(part, ast.Constant) and _CYRILLIC.search(str(part.value))
            for part in node.values
        ):
            continue
        out: list[str] = []
        names = iter(source.params)
        for part in node.values:
            if isinstance(part, ast.Constant):
                out.append(str(part.value))
            else:
                try:
                    out.append("{" + next(names) + "}")
                except StopIteration as exc:
                    raise ExtractError(f"{source.function}: more placeholders than params") from exc
        if next(names, None) is not None:
            raise ExtractError(f"{source.function}: fewer placeholders than params")
        template = "".join(out)
        if source.strip_suffix:
            if not template.endswith(source.strip_suffix):
                raise ExtractError(f"{source.function}: unexpected f-string tail")
            template = template[: -len(source.strip_suffix)]
        return template
    raise ExtractError(f"{source.function}: no f-string with Cyrillic text")


def read_source(src: Path, package: str, source: Source, cache: dict[str, _Module]) -> Any:
    mod = _load(src, package, source.module, cache)
    if isinstance(source, Const):
        if source.name not in mod.consts:
            raise ExtractError(f"{source.module}.{source.name} not found")
        return _eval(mod.consts[source.name], mod)
    if isinstance(source, Inline):
        return _inline(source, mod)
    return _fstring(source, mod)


def extract(src: Path, package: str = "sotto") -> tuple[dict[str, Any], list[str]]:
    """Return ``(flat, problems)``: ``flat`` maps dotted keys to values, ``problems`` lists the
    keys whose cross-check sources disagree (names only)."""
    cache: dict[str, _Module] = {}
    flat: dict[str, Any] = {}
    problems: list[str] = []
    for item in ITEMS:
        value = read_source(src, package, item.source, cache)
        flat[item.key] = value
        for other in item.cross:
            cross = read_source(src, package, other, cache)
            if item.mode == "prefix":
                prefix = str(value).split("{", 1)[0]
                same = cross == prefix
            else:
                same = cross == value
            if not same:
                problems.append(f"{item.key}: disagrees with {other.module}")
    return flat, problems


def nest(flat: dict[str, Any]) -> dict[str, Any]:
    """Dotted keys -> nested mapping in the order of ``ITEMS``."""
    out: dict[str, Any] = {}
    for key, value in flat.items():
        group, leaf = key.split(".", 1)
        out.setdefault(group, {})[leaf] = value
    return out


def dump(nested: dict[str, Any]) -> str:
    return yaml.safe_dump(
        nested, allow_unicode=True, sort_keys=False, width=10**6, default_flow_style=False
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--src", type=Path, required=True, help="directory holding the package")
    parser.add_argument("--package", default="sotto")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        flat, problems = extract(args.src, args.package)
    except ExtractError as exc:
        print(f"extract_fragments: {exc}", file=sys.stderr)
        return 1
    if problems:
        for problem in problems:
            print(f"extract_fragments: {problem}", file=sys.stderr)
        return 2
    args.out.write_text(dump(nest(flat)), encoding="utf-8")
    print(f"extract_fragments: wrote {len(flat)} keys to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
