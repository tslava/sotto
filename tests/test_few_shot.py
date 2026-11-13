"""few_shot.yaml: loading and rendering into the prompt format."""

import pytest
from pydantic import ValidationError

from sotto.few_shot import FewShot, load_few_shot, render_few_shot
from tests.packs import DEMO_PACK

FEW_SHOT_PATH = DEMO_PACK / "few_shot.yaml"


def test_load_real_few_shot_without_errors() -> None:
    items = load_few_shot(FEW_SHOT_PATH)

    assert len(items) == 21
    assert all(isinstance(item, FewShot) for item in items)
    # 5-7 examples must have speak: false, so that silence stays a normal answer.
    assert 5 <= sum(1 for item in items if item.speak is False) <= 7
    assert any(item.speak is True for item in items)


def test_render_two_examples_matches_expected_format() -> None:
    items = [
        FewShot(
            name="Theo",
            user="they moved the meeting again, second time this week",
            speak=True,
            text=(
                "Back in the nineties they put it off, too. Twice. "
                "Then the depot shut and that was that."
            ),
        ),
        FewShot(name="Jo", user="good morning everyone", speak=False),
    ]

    expected = (
        "Theo: they moved the meeting again, second time this week\n"
        '{"speak": true, "text": "Back in the nineties they put it off, too. Twice. '
        'Then the depot shut and that was that."}'
        "\n\n"
        "Jo: good morning everyone\n"
        '{"speak": false, "text": ""}'
    )

    assert render_few_shot(items) == expected


def test_speak_true_requires_non_empty_text() -> None:
    with pytest.raises(ValidationError):
        FewShot(name="X", user="y", speak=True, text="")


def test_speak_false_requires_empty_text() -> None:
    with pytest.raises(ValidationError):
        FewShot(name="X", user="y", speak=False, text="something")
