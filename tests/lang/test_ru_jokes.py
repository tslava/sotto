"""Russian-language behaviour of ``sotto.jokes``."""

from sotto.jokes import joke_used
from tests.packs import RU


def test_joke_used_folds_yo_with_the_russian_language() -> None:
    assert joke_used("Ёлка зелёная", "елка", RU)
