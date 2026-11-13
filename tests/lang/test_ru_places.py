"""Russian-language behaviour of ``sotto.places``."""

from sotto.places import validate_fact
from tests.packs import RU


def test_validate_fact_follows_the_russian_alphabet() -> None:
    assert validate_fact("тихо, дёшево", lang=RU) == "тихо, дёшево"
    assert validate_fact("quiet", lang=RU) is None
