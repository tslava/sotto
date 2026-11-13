"""Загрузка config.yaml с применением плоских overrides из БД (config_overrides)."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args, get_origin

import yaml
from annotated_types import Ge, Gt, Le, Lt
from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo

from sotto.config_models import Config
from sotto.errors import OwnerError
from sotto.i18n import Catalog


class ConfigError(ValueError):
    """``load_config`` found invalid values; ``owner`` is the first error a validator raised as
    ``OwnerError`` (with the key filled in), for the owner UI to render through the catalog."""

    def __init__(self, message: str, owner: OwnerError | None = None) -> None:
        super().__init__(message)
        self.owner = owner


@dataclass(frozen=True)
class KeyInfo:
    """Карточка одного ключа конфига для /get <ключ> (CLAUDE.md, «справка по ключам»)."""

    key: str
    value: str
    default: str
    overridden: bool
    type_name: str
    bounds: str
    description: str
    settable: bool


def _resolve_field_annotation(model: type[BaseModel], parts: list[str], full_key: str) -> Any:
    """Проверяет, что путь parts существует в модели, и возвращает аннотацию листа."""
    field_name = parts[0]
    fields = model.model_fields
    if field_name not in fields:
        raise ValueError(
            f"unknown config key: {full_key!r} (no field {field_name!r} in {model.__name__})"
        )
    annotation = fields[field_name].annotation
    if len(parts) == 1:
        return annotation
    if not (isinstance(annotation, type) and issubclass(annotation, BaseModel)):
        raise ValueError(
            f"unknown config key: {full_key!r} ({'.'.join(parts)!r} is not a config section)"
        )
    return _resolve_field_annotation(annotation, parts[1:], full_key)


def _convert_override_value(annotation: Any, raw_value: str, key: str) -> Any:
    """Строка -> питоновское значение. Списки/bool идут через yaml.safe_load,
    остальное остаётся строкой и приводится типом самим pydantic."""
    origin = get_origin(annotation)
    # dict — ради filters.motifs («метка -> список регулярок», CLAUDE.md, мера 5):
    # /set filters.motifs принимает такой же YAML, как списки регулярок.
    needs_yaml = annotation is bool or origin in (list, tuple, dict)
    if not needs_yaml:
        return raw_value
    try:
        parsed = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"invalid value for override {key!r}: {raw_value!r} is not valid YAML"
        ) from exc
    return parsed


def _set_nested(data: dict[str, Any], parts: list[str], value: Any) -> None:
    node = data
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def _deep_merge(base: dict[str, Any], top: Mapping[str, Any]) -> None:
    """Накатывает top поверх base: словари сливаются по ключам, остальное заменяется целиком."""
    for key, value in top.items():
        current = base.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            _deep_merge(current, value)
        elif isinstance(value, Mapping):
            base[key] = _deep_copy_mapping(value)
        else:
            base[key] = value


def _deep_copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    copy: dict[str, Any] = {}
    _deep_merge(copy, value)
    return copy


def _read_layer(layer: Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(layer, Mapping):
        return layer
    if not layer.exists():
        return {}
    loaded = yaml.safe_load(layer.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config file {layer} must contain a mapping at top level")
    return loaded


def _first_owner_error(errors: Sequence[Mapping[str, Any]]) -> OwnerError | None:
    """The first validator error raised as ``OwnerError``, ``key`` set from the error location."""
    for err in errors:
        original = (err.get("ctx") or {}).get("error")
        if isinstance(original, OwnerError):
            params = dict(original.params)
            params.setdefault("key", ".".join(str(part) for part in err["loc"]))
            return OwnerError(original.code, **params)
    return None


def load_config(
    layers: Path | Mapping[str, Any] | Sequence[Path | Mapping[str, Any]],
    overrides: dict[str, str] | None = None,
) -> Config:
    """Читает слои конфига, накатывает плоские overrides ("section.key" -> "value") и валидирует.

    layers — один слой или последовательность от нижнего к верхнему (yaml-файл или готовый
    словарь); слои сливаются вглубь, списки и скаляры верхнего слоя заменяют нижние.
    Отсутствующий файл — пустой слой. Невалидный override или неизвестный ключ ->
    ValueError с ключом в тексте.
    """
    sequence: Sequence[Path | Mapping[str, Any]] = (
        [layers] if isinstance(layers, (Path, Mapping)) else layers
    )
    data: dict[str, Any] = {}
    for layer in sequence:
        _deep_merge(data, _read_layer(layer))

    if overrides:
        for key, raw_value in overrides.items():
            parts = key.split(".")
            annotation = _resolve_field_annotation(Config, parts, key)
            value = _convert_override_value(annotation, raw_value, key)
            _set_nested(data, parts, value)

    try:
        return Config.model_validate(data)
    except ValidationError as exc:
        errors = exc.errors()
        lines = [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in errors]
        raise ConfigError(
            "invalid config: " + "; ".join(lines), _first_owner_error(errors)
        ) from exc


def _format_value(value: object) -> str:
    """Питоновское значение листового поля -> строка, единый формат для /get и /set."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    if isinstance(value, (list, tuple)):
        plain = [
            item.model_dump(mode="json") if isinstance(item, BaseModel) else item for item in value
        ]
        return yaml.safe_dump(
            plain, default_flow_style=True, allow_unicode=True, width=10_000
        ).strip()
    if isinstance(value, dict):
        # filters.motifs: тот же YAML-формат, что и списки, чтобы /get отдавал
        # ровно то, что принимает /set.
        return yaml.safe_dump(
            value, default_flow_style=True, allow_unicode=True, width=10_000
        ).strip()
    return str(value)


def flatten_config(cfg: Config) -> dict[str, str]:
    """Config -> плоский словарь "section.key" -> строка, для будущей команды /get."""
    result: dict[str, str] = {}

    def _walk(prefix: str, obj: Any) -> None:
        if isinstance(obj, BaseModel):
            for name in obj.__class__.model_fields:
                child = getattr(obj, name)
                _walk(f"{prefix}.{name}" if prefix else name, child)
        else:
            result[prefix] = _format_value(obj)

    _walk("", cfg)
    return result


def _type_name(annotation: Any) -> str:
    """Аннотация поля -> короткое имя типа для карточки /get <ключ>."""
    origin = get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type):
            return annotation.__name__
        return str(annotation)
    args = get_args(annotation)
    if origin is list:
        inner = _type_name(args[0]) if args else "Any"
        return f"list[{inner}]"
    if origin is tuple:
        inner = ", ".join(_type_name(arg) for arg in args)
        return f"tuple[{inner}]"
    if origin is dict:
        inner = ", ".join(_type_name(arg) for arg in args) if args else "Any, Any"
        return f"dict[{inner}]"
    return str(origin)


def _format_bound(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _bounds_str(field_info: FieldInfo) -> str:
    """Границы поля из annotated_types-метаданных Field(ge=..., le=...).

    "0..50" при заданных обеих границах, ">=1"/"<=5" при одной, "" без границ.
    """
    ge = gt = le = lt = None
    for constraint in field_info.metadata:
        if isinstance(constraint, Ge):
            ge = constraint.ge
        elif isinstance(constraint, Gt):
            gt = constraint.gt
        elif isinstance(constraint, Le):
            le = constraint.le
        elif isinstance(constraint, Lt):
            lt = constraint.lt
    lo = ge if ge is not None else gt
    hi = le if le is not None else lt
    if lo is not None and hi is not None:
        return f"{_format_bound(lo)}..{_format_bound(hi)}"
    if lo is not None:
        return f">={_format_bound(lo)}"
    if hi is not None:
        return f"<={_format_bound(hi)}"
    return ""


def _description(field_info: FieldInfo, key: str, catalog: Catalog | None) -> str:
    catalog_key = f"config_keys.{key}"
    if catalog is not None and catalog.has(catalog_key):
        return catalog.t(catalog_key)
    return field_info.description or ""


def describe_key(
    current: Config,
    base: Config,
    overrides: dict[str, str],
    key: str,
    catalog: Catalog | None = None,
) -> KeyInfo | None:
    """Карточка одного ключа конфига: тип, границы, описание, текущее и дефолтное значение.

    Описание берётся из каталога (``config_keys.<ключ>``, свой язык пака или английский
    запасной), а при его отсутствии или без каталога — из ``Field.description`` модели.

    None, если такого ключа нет, либо key указывает на секцию (BaseModel), а не на лист
    (обход по model_fields, как в flatten_config).
    """
    parts = key.split(".")
    model_cls: type[BaseModel] = Config
    field_info: FieldInfo | None = None
    for index, part in enumerate(parts):
        fields = model_cls.model_fields
        if part not in fields:
            return None
        field_info = fields[part]
        annotation = field_info.annotation
        is_last = index == len(parts) - 1
        if is_last:
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                return None
            break
        if not (isinstance(annotation, type) and issubclass(annotation, BaseModel)):
            return None
        model_cls = annotation

    if field_info is None:
        return None

    current_value: Any = current
    default_value: Any = base
    for part in parts:
        current_value = getattr(current_value, part)
        default_value = getattr(default_value, part)

    return KeyInfo(
        key=key,
        value=_format_value(current_value),
        default=_format_value(default_value),
        overridden=key in overrides,
        type_name=_type_name(field_info.annotation),
        bounds=_bounds_str(field_info),
        description=_description(field_info, key, catalog),
        settable=not key.startswith("persona."),
    )
