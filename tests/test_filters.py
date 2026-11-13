"""Output filter (layers 1-2 and check_output assembly) on the English demo pack.

A table drives the text through ``check_output`` (no judge) and expects either a full pass or
a given reason among ``verdict.reasons``. The Russian-specific checks (foreign-script venue and
Latin filters, Cyrillic folding, Russian lists) live in ``tests/lang/test_ru_filters.py``.
"""

from __future__ import annotations

import pytest

from sotto.config_models import Config
from sotto.db import MessageRow
from sotto.few_shot import load_few_shot
from sotto.filters import FilterContext, FilterVerdict, check_output
from tests.demo import DEMO_FRAG, DEMO_PACK, EN, demo_config

SYSTEM_PROMPT = (DEMO_PACK / "prompts" / "system.txt").read_text(encoding="utf-8")


def _row(display_name: str, text: str, *, is_bot: bool = False, created_at: int = 1) -> MessageRow:
    return MessageRow(
        id=created_at,
        tg_message_id=created_at,
        chat_id=1,
        user_id=1,
        display_name=display_name,
        text=text,
        reply_to_tg_message_id=None,
        is_bot=is_bot,
        created_at=created_at,
    )


def _ctx(
    *,
    cfg: Config | None = None,
    recent_replies: list[str] | None = None,
    context_rows: list[MessageRow] | None = None,
    places_names: list[str] | None = None,
    participant_names: list[str] | None = None,
    bot_names: list[str] | None = None,
    muted_names: list[str] | None = None,
    system_prompt: str = "",
    trigger_text: str = "",
    now: int = 0,
) -> FilterContext:
    return FilterContext(
        lang=EN,
        cfg=cfg if cfg is not None else demo_config(),
        recent_replies=recent_replies or [],
        context_rows=context_rows or [],
        places_names=places_names or [],
        participant_names=participant_names or [],
        bot_names=bot_names or [],
        muted_names=muted_names or [],
        system_prompt=system_prompt,
        trigger_text=trigger_text,
        now=now,
        frag=DEMO_FRAG,
    )


_DEFAULT_CTX = _ctx()

# (id, text, ctx, expected_reason_or_None): None means a full pass, a string means ok=False
# with that reason among verdict.reasons.
CASES: list[tuple[str, str, FilterContext, str | None]] = [
    # --- voice: length / markup / emoji / sentences ---
    ("voice_pass_coffee", "The coffee is fine, I will not argue.", _DEFAULT_CTX, None),
    (
        "sentences_three_full_cut",
        "I drove to the shop early this morning. I bought bread and milk there. "
        "In the evening I will fry some eggs.",
        _DEFAULT_CTX,
        "regex:sentences",
    ),
    # a fragment counts as a sentence from 3 words: a chopped tale passes
    (
        "sentences_chopped_tale_pass",
        "Back in the nineties they put it off, too. Twice. Then the depot shut and that was that.",
        _DEFAULT_CTX,
        None,
    ),
    ("sentences_two_short_pass", "Right, right. Tea then?", _DEFAULT_CTX, None),
    (
        "markdown_bullet",
        "Here is the plan:\n- go to the shop\n- buy bread",
        _DEFAULT_CTX,
        "regex:markdown",
    ),
    ("markdown_numbered", "1. First thing\n2. Second thing", _DEFAULT_CTX, "regex:markdown"),
    ("markdown_bold", "This is **really** important, trust me.", _DEFAULT_CTX, "regex:markdown"),
    ("markdown_header", "# Important: do not open the door.", _DEFAULT_CTX, "regex:markdown"),
    ("markdown_codeblock", "```some code``` do not open it.", _DEFAULT_CTX, "regex:markdown"),
    ("emoji", "Hello everyone today 🚀 fine.", _DEFAULT_CTX, "regex:emoji"),
    ("length_301", "y" * 301, _DEFAULT_CTX, "regex:length"),
    ("length_boundary_300_pass", "y" * 300, _DEFAULT_CTX, None),
    # --- allowed emoji ---
    ("emoji_allowed_end_pass", "It happens 🙂", _DEFAULT_CTX, None),
    ("emoji_allowed_after_dot_pass", "It happens. 👍", _DEFAULT_CTX, None),
    ("emoji_allowed_thumbsup_skin_tone_pass", "It happens 👍🏻", _DEFAULT_CTX, None),
    ("emoji_count_two_cut", "It happens 🙂🙂", _DEFAULT_CTX, "style:emoji_count"),
    ("emoji_start_pass", "🙂 It happens", _DEFAULT_CTX, None),
    ("emoji_disallowed_cut", "It happens 🚀", _DEFAULT_CTX, "regex:emoji"),
    (
        "emoji_freq_recent_emoji_cut",
        "It happens 🙂",
        _ctx(recent_replies=["Everyone went home.", "It happens.", "Fine.", "Pity 😂"]),
        "style:emoji_freq",
    ),
    (
        "emoji_freq_old_emoji_pass",
        "It happens 🙂",
        _ctx(
            recent_replies=[
                "Pity 😂",
                "Everyone went home.",
                "Fine, I am in the shed.",
                "Bought peat for the seedlings.",
                "Good night all.",
            ]
        ),
        None,
    ),
    ("emoji_none_no_style_triggers", "It happens, to everyone.", _DEFAULT_CTX, None),
    # --- starts with a name ---
    (
        "starts_name_participant_cut",
        "Dana, you are wrong.",
        _ctx(participant_names=["Dana"]),
        "regex:starts_name",
    ),
    (
        "starts_name_bot_name_cut",
        "Gus, you are late as usual.",
        _ctx(bot_names=["Gus"]),
        "regex:starts_name",
    ),
    (
        "starts_name_not_first_word_pass",
        "Well you do go on, that is something.",
        _ctx(participant_names=["Dana"]),
        None,
    ),
    # --- phone ---
    (
        "phone_with_plus",
        "Here is his number +44 7700 900 123, go ahead and call.",
        _DEFAULT_CTX,
        "regex:phone",
    ),
    ("phone_pass_year", "That was in 2020, before the move.", _DEFAULT_CTX, None),
    # --- topic stop list on output ---
    (
        "topic_stop_cut",
        "All right, moving on, but the elections are a mess.",
        _DEFAULT_CTX,
        "regex:topic",
    ),
    ("topic_pass_peat", "Bought peat for the seedlings at the weekend.", _DEFAULT_CTX, None),
    # --- muted ---
    (
        "muted_name_cut",
        "Theo is winding everyone up again, what a laugh.",
        _ctx(muted_names=["Theo"]),
        "regex:muted_name",
    ),
    (
        "muted_name_absent_pass",
        "Everyone went home rather quickly.",
        _ctx(muted_names=["Theo"]),
        None,
    ),
    # --- echo ---
    (
        "echo_four_words_cut",
        "I know a decent electrician in town, but I never called him.",
        _ctx(context_rows=[_row("Jo", "can anyone recommend a decent electrician in town")]),
        "regex:echo",
    ),
    (
        "echo_three_words_pass",
        "I am also looking for a decent electrician, no luck.",
        _ctx(context_rows=[_row("Jo", "a decent electrician in town wanted")]),
        None,
    ),
    # --- model talk, assistant markers, grumpy ---
    (
        "model_talk_cut",
        "I am answering you as a language model, no frills.",
        _DEFAULT_CTX,
        "regex:model_talk",
    ),
    ("model_talk_pass", "I will answer you, but not today.", _DEFAULT_CTX, None),
    (
        "style_assistant",
        "Certainly! You will sort it out yourself.",
        _DEFAULT_CTX,
        "style:assistant",
    ),
    ("style_assistant_pass", "I would not, but it is your call.", _DEFAULT_CTX, None),
    (
        "style_grumpy_told_you",
        "It was on my sister's side. I told you that already.",
        _DEFAULT_CTX,
        "style:grumpy",
    ),
    (
        "style_grumpy_nobody_cares",
        "I will not name the place, nobody cares anyway",
        _DEFAULT_CTX,
        "style:grumpy",
    ),
    (
        "style_grumpy_warm_pass",
        "Married thirty years. My sister is still surprised.",
        _DEFAULT_CTX,
        None,
    ),
    # --- dedup, layer 2 (Jaccard) ---
    (
        "dedup_jaccard_cut",
        "Right, he does not talk on the phone, you have to catch him, that sort of character.",
        _ctx(
            recent_replies=[
                "Right. He does not talk on the phone, you have to catch him, that sort of man."
            ]
        ),
        "dedup:jaccard",
    ),
    (
        "dedup_jaccard_different_pass",
        "I could not say, but he is a decent sort on the whole.",
        _ctx(
            recent_replies=[
                "Right. He does not talk on the phone, you have to catch him, that sort of man."
            ]
        ),
        None,
    ),
    ("dedup_exact_short_cut", "It happens.", _ctx(recent_replies=["It happens."]), "dedup:jaccard"),
    ("dedup_short_different_pass", "Pity.", _ctx(recent_replies=["It happens."]), None),
    # --- self echo ---
    (
        "self_echo_repeated_gram_cut",
        "went fishing early yesterday morning",
        _ctx(recent_replies=["went fishing early yesterday morning too"]),
        "dedup:self_echo",
    ),
    (
        "self_echo_stopwords_pass",
        "It is not that, so I said.",
        _ctx(recent_replies=["Yes, it is not that, no."]),
        None,
    ),
    # --- question twice, exclaim ---
    (
        "question_x2_cut",
        "What do you think?",
        _ctx(recent_replies=["Are you serious?"]),
        "style:question_x2",
    ),
    (
        "question_after_statement_pass",
        "What do you think?",
        _ctx(recent_replies=["It happens."]),
        None,
    ),
    ("style_exclaim", "Well you do go on!! Fancy that.", _DEFAULT_CTX, "style:exclaim"),
    ("style_exclaim_pass", "Oh well, never mind!", _DEFAULT_CTX, None),
]


@pytest.mark.parametrize("case_id, text, ctx, expected", CASES, ids=[c[0] for c in CASES])
async def test_check_output_table(
    case_id: str, text: str, ctx: FilterContext, expected: str | None
) -> None:
    verdict = await check_output(text, ctx)
    if expected is None:
        assert verdict == FilterVerdict(ok=True, reason="pass", reasons=())
    else:
        assert verdict.ok is False
        assert expected in verdict.reasons
        assert verdict.reason == verdict.reasons[0]


async def test_check_output_collects_all_matching_reasons() -> None:
    text = "- Do it this way\n- And this way too 🚀"
    verdict = await check_output(text, _DEFAULT_CTX)
    assert verdict.ok is False
    assert verdict.reasons == ("regex:markdown", "regex:emoji")
    assert verdict.reason == "regex:markdown"


# --- the judge layer ---


class _FakeJudge:
    def __init__(self, reasons: list[str]) -> None:
        self.called = False
        self._reasons = reasons

    async def check(self, *, candidate: str, trigger_text: str, now: int) -> list[str]:
        self.called = True
        return list(self._reasons)


async def test_judge_called_when_layers_clean() -> None:
    judge = _FakeJudge([])
    verdict = await check_output("The coffee is fine, I will not argue.", _DEFAULT_CTX, judge=judge)
    assert judge.called is True
    assert verdict.ok is True


async def test_judge_not_called_when_dirty_and_shadow_false() -> None:
    cfg = demo_config()
    cfg.filters.shadow = False
    judge = _FakeJudge(["judge:risky"])
    verdict = await check_output("y" * 301, _ctx(cfg=cfg), judge=judge)
    assert judge.called is False
    assert verdict.ok is False
    assert verdict.reason == "regex:length"
    assert "judge:risky" not in verdict.reasons


async def test_judge_called_and_reasons_added_when_dirty_and_shadow_true() -> None:
    cfg = demo_config()
    cfg.filters.shadow = True
    judge = _FakeJudge(["judge:risky"])
    verdict = await check_output("y" * 301, _ctx(cfg=cfg), judge=judge)
    assert judge.called is True
    assert verdict.ok is False
    assert verdict.reason == "regex:length"
    assert "judge:risky" in verdict.reasons


async def test_filter_context_accepts_the_four_field_call() -> None:
    ctx = FilterContext(
        cfg=demo_config(),
        recent_replies=[],
        context_rows=[],
        places_names=[],
        frag=DEMO_FRAG,
        lang=EN,
    )
    verdict = await check_output("The coffee is fine, I will not argue.", ctx)
    assert verdict.ok is True
    assert ctx.participant_names == []
    assert ctx.bot_names == []
    assert ctx.muted_names == []
    assert ctx.patterns is None
    assert ctx.system_prompt == ""
    assert ctx.trigger_text == ""
    assert ctx.now == 0


# --- prompt leak: only the instruction part counts, not the biography ---


async def test_prompt_leak_biography_paraphrase_passes() -> None:
    prompt = "- You never give advice about the job and you never explain how things work.\n"
    ctx = _ctx(system_prompt=prompt + "You drink little, one pint of something cheap and brown.")
    verdict = await check_output("You drink little, one pint of something cheap, enough.", ctx)
    assert "regex:prompt_leak" not in verdict.reasons


async def test_prompt_leak_instruction_bullet_is_cut() -> None:
    prompt = "- You never give advice about the job and you never explain how things work.\n"
    ctx = _ctx(system_prompt=prompt)
    verdict = await check_output("You never give advice about the job, full stop.", ctx)
    assert "regex:prompt_leak" in verdict.reasons


async def test_all_few_shot_speak_true_examples_pass_check_output() -> None:
    items = load_few_shot(DEMO_PACK / "few_shot.yaml")
    speak_items = [item for item in items if item.speak]
    assert speak_items, "few_shot.yaml needs at least one speak=true example"

    cfg = demo_config()
    participant_names = [item.name for item in items]
    bot_names = [cfg.persona.name, cfg.persona.display_name, *cfg.persona.name_triggers]

    failures: list[str] = []
    for item in speak_items:
        ctx = _ctx(
            cfg=cfg,
            participant_names=participant_names,
            bot_names=bot_names,
            system_prompt=SYSTEM_PROMPT,
            trigger_text=item.user,
        )
        verdict = await check_output(item.text, ctx)
        if not verdict.ok:
            failures.append(f"{item.name!r}: {item.text!r} -> {verdict.reasons}")

    assert not failures, "few_shot.yaml examples cut by the output filter:\n" + "\n".join(failures)


# --- motifs and stories ---


async def test_dedup_motif_cuts_repeated_prop() -> None:
    ctx = _ctx(recent_replies=["The clocks again, my sister says enough.", "We went quiet."])
    verdict = await check_output("Another clock came in this morning.", ctx)
    assert "dedup:motif" in verdict.reasons


async def test_dedup_motif_allows_other_prop() -> None:
    ctx = _ctx(recent_replies=["The clocks again, my sister says enough."])
    verdict = await check_output("The tomatoes are coming on, too many of them.", ctx)
    assert "dedup:motif" not in verdict.reasons


async def test_dedup_motif_ignores_replies_outside_window() -> None:
    cfg = demo_config()
    cfg.filters.motif_recent_window = 3
    ctx = _ctx(
        cfg=cfg,
        recent_replies=[
            "The tuba is loud.",
            "We went quiet.",
            "The keys turned up.",
            "Got a jumper.",
        ],
    )
    verdict = await check_output("The tuba needs a polish again.", ctx)
    assert "dedup:motif" not in verdict.reasons


async def test_dedup_motif_disabled_by_zero_window() -> None:
    cfg = demo_config()
    cfg.filters.motif_recent_window = 0
    ctx = _ctx(cfg=cfg, recent_replies=["The tuba is loud."])
    verdict = await check_output("The tuba needs a polish again.", ctx)
    assert "dedup:motif" not in verdict.reasons


async def test_style_story_quota_cuts_second_story_in_window() -> None:
    ctx = _ctx(recent_replies=["I remember it being just the same."])
    verdict = await check_output("One time we went through all this already.", ctx)
    assert "style:story_quota" in verdict.reasons


async def test_style_story_quota_allows_short_answer_without_marker() -> None:
    ctx = _ctx(recent_replies=["I remember it being just the same."])
    verdict = await check_output("Regularly. I just stopped getting upset.", ctx)
    assert "style:story_quota" not in verdict.reasons


async def test_style_story_quota_allows_first_story_in_window() -> None:
    ctx = _ctx(recent_replies=["We went quiet.", "The keys turned up."])
    verdict = await check_output("One time we went through all this already.", ctx)
    assert "style:story_quota" not in verdict.reasons


async def test_style_story_quota_zero_max_cuts_any_story() -> None:
    cfg = demo_config()
    cfg.filters.story_max = 0
    verdict = await check_output("One time we went through all this already.", _ctx(cfg=cfg))
    assert "style:story_quota" in verdict.reasons


async def test_motifs_and_stories_ignore_sticker_records() -> None:
    ctx = _ctx(recent_replies=["[sticker #2] The clocks again", "[sticker #3] I remember that"])
    verdict = await check_output("In the shed the clocks, I remember, once ticked.", ctx)
    assert "dedup:motif" not in verdict.reasons
    assert "style:story_quota" not in verdict.reasons


async def test_english_has_no_foreign_script_checks() -> None:
    ctx = _ctx(recent_replies=["Some Foreign Words"])
    verdict = await check_output("Szalony Warsztat dobry", ctx)
    assert not any(r in verdict.reasons for r in ("regex:venue", "regex:latin"))
    assert "dedup:polish_freq" not in verdict.reasons


async def test_missing_patterns_and_lang_fall_back_to_english() -> None:
    ctx = FilterContext(
        cfg=Config(), recent_replies=[], context_rows=[], places_names=[], frag=DEMO_FRAG
    )
    verdict = await check_output("Szalony Warsztat dobry", ctx)
    assert verdict.ok
