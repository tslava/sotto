"""The engine takes its wording from the pack's fragments, not from its own constants.

Each test swaps one fragment leaf for a synthetic one and checks that the rendered text, the
persisted placeholder or the derived regex follows. The legacy fragments are the control.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import yaml

from sotto import motifs, postprocess
from sotto.chat_memory import render_chat_memory
from sotto.config import load_config
from sotto.config_models import Config, FiltersConfig, PlacesConfig, VocabConfig, WeatherConfig
from sotto.db import (
    ChatMemoryRow,
    JokeRow,
    LifeEventRow,
    MessageRow,
    PlaceRow,
    SelfFactRow,
)
from sotto.diary import render_diary
from sotto.export_parser import parse_export
from sotto.followup import FollowupChecker
from sotto.jokes import render_jokes
from sotto.judge import Judge
from sotto.llm import LLMResult
from sotto.places import render_places_block, render_places_menu, select_places
from sotto.places_fill import (
    compress_fact,
    pick_district,
    search_places,
)
from sotto.prompt import (
    build_messages,
    render_life,
    situation_addressed,
    situation_callback,
    situation_checkin,
    situation_followup,
    situation_life,
)
from sotto.reactions import REPLY_UNAVAILABLE, ReactionChooser
from sotto.sanitize import media_placeholder, sanitize_display_name, stable_n
from sotto.stickers import (
    Sticker,
    recent_sticker_ids,
    render_sticker_menu,
    sticker_tag_re,
)
from sotto.stickers_fill import recognize_sticker
from sotto.vision import VisionDescriber, photo_text
from sotto.vocab import VocabDict, VocabWord, offer_allowed, pick_offer, render_vocab
from sotto.weather import (
    Place,
    Weather,
    WeatherClient,
    describe,
    render_weather,
    render_weather_place,
)
from tests.packs import LEGACY_PACK, RU, frag_with, legacy_config, ru_vocab

NOW = 1_767_268_800  # 2026-01-01 12:00 UTC


# -- persisted placeholders ----------------------------------------------------------


def test_participant_fallback_follows_the_pack() -> None:
    frag = frag_with(placeholders__participant="Guest {n}")

    assert sanitize_display_name("", 5, set(), frag=frag) == f"Guest {stable_n(5)}"
    assert sanitize_display_name("Bob", 5, {"bob"}, frag=frag) == f"Guest {stable_n(5)}"
    assert sanitize_display_name("Anna", 5, set(), frag=frag) == "Anna"
    assert frag.prefix("placeholders.participant") == "Guest "


@pytest.mark.parametrize(
    ("field", "key"),
    [
        ("sticker", "sticker"),
        ("photo", "photo"),
        ("voice", "voice"),
        ("audio", "voice"),
        ("video", "video"),
        ("video_note", "video"),
        ("animation", "video"),
        ("document", "file"),
    ],
)
def test_media_placeholders_follow_the_pack(field: str, key: str) -> None:
    frag = frag_with(**{f"placeholders__{key}": f"<{key}>"})
    names = ("sticker", "photo", "voice", "audio", "video", "video_note", "animation", "document")
    message = SimpleNamespace(**{name: (object() if name == field else None) for name in names})

    assert media_placeholder(message, frag=frag) == f"<{key}>"
    assert media_placeholder(SimpleNamespace(**dict.fromkeys(names)), frag=frag) is None


def test_export_parser_uses_the_pack_placeholders(tmp_path: Any) -> None:
    import json

    frag = frag_with(
        placeholders__photo="<p>",
        placeholders__sticker="<s>",
        placeholders__voice="<v>",
        placeholders__video="<m>",
        placeholders__file="<f>",
    )
    media = [
        {"photo": "photos/1.jpg"},
        {"media_type": "sticker"},
        {"media_type": "voice_message"},
        {"media_type": "audio_file"},
        {"media_type": "video_file"},
        {"file": "files/1.bin"},
    ]
    messages = [
        {
            "id": index,
            "type": "message",
            "date": "2026-01-01T12:00:00",
            "from": "Anna",
            "from_id": "user1",
            "text": "",
            **extra,
        }
        for index, extra in enumerate(media, start=1)
    ]
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"messages": messages}), encoding="utf-8")

    parsed = parse_export(path, "UTC", frag=frag)

    assert [m.text for m in parsed] == ["<p>", "<s>", "<v>", "<v>", "<m>", "<f>"]


def test_photo_text_follows_the_pack() -> None:
    frag = frag_with(placeholders__photo="<p>", placeholders__photo_with_desc="<p: {desc}>")

    assert photo_text(None, "", frag=frag) == "<p>"
    assert photo_text("a cat", "hello", frag=frag) == "<p: a cat> hello"


def test_sticker_tag_regex_and_prefix_are_derived_from_the_pack() -> None:
    frag = frag_with(placeholders__sticker_tag="<<sticker {id}>>")

    pattern = sticker_tag_re(frag)

    assert pattern.match("<<sticker 12>> caption") is not None
    assert pattern.match("[sticker 12] caption") is None
    texts = ["plain", "<<sticker 3>> a", "<<sticker 4>> b", "other"]
    assert recent_sticker_ids(texts, 5, frag=frag) == {3, 4}
    assert recent_sticker_ids(texts, 1, frag=frag) == {4}
    assert frag.prefix("placeholders.sticker_tag") == "<<sticker "
    assert isinstance(pattern, re.Pattern)


def test_sticker_records_are_skipped_by_prefix_in_motifs_postprocess_and_vocab() -> None:
    frag = frag_with(placeholders__sticker_tag="<<sticker {id}>>")
    marker = [re.compile("wife")]
    replies = ["<<sticker 1>> wife here", "nothing"]

    assert motifs.used_motifs(replies, 5, {"wife": marker}, frag=frag) == []
    assert motifs.used_motifs(["[sticker 1] wife"], 5, {"wife": marker}, frag=frag) == ["wife"]
    assert motifs.story_count(replies, 5, marker, frag=frag) == 0
    assert motifs.speaks_about_self("<<sticker 1>> wife", marker, frag=frag) is False
    assert motifs.speaks_about_self("wife", marker, frag=frag) is True
    share = motifs.self_share([("ambient", "<<sticker 1>> wife")], marker, frag=frag)
    assert share.ambient == (0, 0)

    cfg = FiltersConfig(emoji_recent_window=2)
    fixed = postprocess.soften(
        "ok \U0001f44d", recent_replies=["<<sticker 7>> \U0001f44d"], cfg=cfg, frag=frag
    )
    assert fixed.fixes == ()
    fixed = postprocess.soften(
        "ok \U0001f44d", recent_replies=["[sticker 7] \U0001f44d"], cfg=cfg, frag=frag
    )
    assert fixed.fixes == ("fix:emoji_freq",)

    vocab = ru_vocab([VocabDict(name="d", words=[VocabWord(word="zork", meaning="m")])])
    vocab_cfg = VocabConfig(probability=1.0, recent_window=3)
    kwargs = {"cfg": vocab_cfg, "vocab": vocab, "count_today": 0, "rng": random.Random(1)}
    assert offer_allowed(recent_replies=["<<sticker 1>> zork"], frag=frag, **kwargs)
    assert not offer_allowed(recent_replies=["[sticker 1] zork"], frag=frag, **kwargs)
    only_word = pick_offer(
        vocab, recent_replies=["<<sticker 1>> zork"], size=1, rng=random.Random(1), frag=frag
    )
    assert [entry.word for entry in only_word] == ["zork"]


# -- situations and the user message -------------------------------------------------


def test_situations_follow_the_pack() -> None:
    frag = frag_with(
        situations__addressed_single="S1 {name}={text}",
        situations__addressed_multi_header="MH",
        situations__addressed_multi_footer="MF",
        situations__followup_single="F1 {name}={text}",
        situations__followup_multi_header="FH",
        situations__followup_multi_footer="FF",
        situations__checkin_header="CH",
        situations__checkin_footer="CF",
        situations__life="L {text}",
        situations__callback="C {name}/{text}",
        render__quote_open="<",
        render__quote_close=">",
    )
    items = [("Ann", "hi"), ("Bob", "yo")]
    row = MessageRow(1, 1, 1, 1, "Ann", "hello", None, False, NOW)

    assert situation_addressed(items[:1], frag=frag) == "S1 Ann=hi"
    assert situation_addressed(items, frag=frag) == "MH\n- Ann: <hi>\n- Bob: <yo>\nMF"
    assert situation_followup(items[:1], frag=frag) == "F1 Ann=hi"
    assert situation_followup(items, frag=frag) == "FH\n- Ann: <hi>\n- Bob: <yo>\nFF"
    assert situation_checkin([row], frag=frag) == "CH\n1. Ann: hello\nCF"
    assert situation_life("news", frag=frag) == "L news"
    assert situation_callback("Ann", "left", frag=frag) == "C Ann/left"
    assert situation_addressed([], frag=frag) == ""


def test_build_messages_uses_pack_scaffolding() -> None:
    frag = frag_with(
        prompt__data_notice="DATA",
        prompt__recent_replies_label="RECENT",
        prompt__context_marker="CTX",
        prompt__recent_replies_marker="REC",
        prompt__places_marker="PLC",
        prompt__places_none="NOPLACES",
        prompt__json_reminder="JSON1",
        placeholders__recent_replies_empty="NONE",
    )
    template = "{context}|{recent_replies}|{places}|{age}|{few_shot}|{situation}"

    system, user = build_messages(
        template,
        age=3,
        few_shot="fs",
        context="c",
        recent_replies="",
        places="",
        situation="",
        frag=frag,
    )

    assert system["content"] == "CTX|REC|PLC|3|fs|"
    lines = user["content"].split("\n")
    assert lines[0] == "DATA" and "RECENT" in lines and "NOPLACES" in lines
    assert "NONE" in lines and lines[-1] == "JSON1"
    _, custom = build_messages(
        template,
        age=3,
        few_shot="",
        context="c",
        recent_replies="r",
        places="p",
        situation="s",
        json_reminder="OWN",
        frag=frag,
    )
    assert custom["content"].split("\n")[-1] == "OWN"


# -- headers of the system-prompt blocks ---------------------------------------------


def test_block_headers_follow_the_pack() -> None:
    frag = frag_with(
        render__life_header="LH",
        render__jokes_header="JH: ",
        render__chat_memory_header="MH",
        render__diary_today_header="DT: ",
        render__diary_week_header="DW: ",
        render__quote_open="'",
        render__quote_close="'",
    )
    life = LifeEventRow(1, "bought a boat", NOW, None, None)
    assert render_life([life], "UTC", frag=frag).split("\n")[0] == "LH"

    joke = JokeRow(1, "knock knock", NOW, None, 0)
    assert render_jokes([joke], now=NOW, cooldown_days=1, frag=frag) == "JH: 'knock knock'."

    memory = ChatMemoryRow(1, NOW - 86400 * 7, NOW, "line one", NOW)
    assert render_chat_memory([memory], "UTC", frag=frag).split("\n")[0] == "MH"

    today = SelfFactRow(1, "went fishing", None, NOW)
    week = SelfFactRow(2, "fixed a roof", None, NOW - 3 * 86400)
    text = render_diary([today], [week], "UTC", frag=frag)
    assert text.startswith("DT: 12:00 went fishing.")
    assert "DW: " in text and "fixed a roof." in text


def test_diary_weekday_names_come_from_the_pack() -> None:
    names = ["d1", "d2", "d3", "d4", "d5", "d6", "d7"]
    frag = frag_with(render__weekdays=names, render__diary_week_header="W: ")
    day = SelfFactRow(1, "x", None, NOW)  # 2026-01-01 is a Thursday

    assert render_diary([], [day], "UTC", frag=frag) == "W: d4 x."


def test_motif_avoid_and_self_share_follow_the_pack() -> None:
    frag = frag_with(
        render__avoid_template="AVOID {items}",
        render__avoid_no_story="NOSTORY",
        render__accusative={"wife": "WIFE-ACC"},
        render__self_share="{days}d a={ambient_hit}/{ambient_total} "
        "b={address_hit}/{address_total} c={other_hit}/{other_total}",
    )

    assert motifs.render_avoid(["wife", "unknown"], no_story=True, frag=frag) == (
        "AVOID WIFE-ACC, unknown NOSTORY"
    )
    assert motifs.render_avoid([], no_story=False, frag=frag) == ""
    share = motifs.SelfShare(ambient=(1, 2), address=(3, 4), other=(5, 6))
    assert motifs.render_self_share(share, 3, frag=frag) == "3d a=1/2 b=3/4 c=5/6"


def test_vocab_headers_follow_the_pack() -> None:
    frag = frag_with(
        vocab__offer_header="OH",
        vocab__offer_tail=": OT ",
        vocab__request_header="RH",
        vocab__request_tail=": RT ",
        vocab__meanings_header="MH: ",
        render__quote_open="<",
        render__quote_close=">",
    )
    vocab = ru_vocab(
        [
            VocabDict(
                name="d",
                about="about",
                words=[VocabWord(word="zork", meaning="Thing", example="a zork")],
            )
        ]
    )
    entry = vocab.entries[0]

    assert render_vocab([entry], [], frag=frag) == "OH (about): OT zork (thing; <a zork>)."
    assert (
        render_vocab([entry], [], requested=True, frag=frag)
        == "RH (about): RT zork (thing; <a zork>)."
    )
    assert render_vocab([], [entry], frag=frag) == "MH: zork: thing."
    assert render_vocab([], [], frag=frag) == ""


def test_places_blocks_follow_the_pack() -> None:
    frag = frag_with(
        prompt__places_none="NONE",
        render__places_block_header="BLOCK",
        render__places_menu_header="MENU",
        render__places_quiet="calm",
        render__places_noisy="loud",
    )
    rows = [
        PlaceRow("b", "Beta", "North", "pub", 4.5, 100, None, True, "dim", True, 1),
        PlaceRow("a", "Alpha", "South", "pub", 4.5, 100, None, False, "", True, 1),
    ]

    assert render_places_menu([], frag=frag) == "NONE" == render_places_block([], frag=frag)
    assert render_places_menu(rows, frag=frag).split("\n") == [
        "MENU",
        "- Alpha, South, pub, loud",
        "- Beta, North, pub, calm, dim",
    ]
    assert render_places_block(rows, frag=frag).split("\n")[0] == "BLOCK"


def test_sticker_menu_uses_pack_quotes() -> None:
    frag = frag_with(render__quote_open="<", render__quote_close=">")
    stickers = [
        Sticker(id=1, file_id="f", text="hey", when="greeting"),
        Sticker(id=2, file_id="g", text="bye"),
        Sticker(id=3, file_id="h", text="off", enabled=False),
    ]

    assert render_sticker_menu(stickers, frag=frag) == "1: <hey> — greeting\n2: <bye>"


# -- weather -------------------------------------------------------------------------


def _weather(**changes: Any) -> Weather:
    values: dict[str, Any] = {
        "temp_now": 9.0,
        "code_now": 0,
        "today_min": 2.0,
        "today_max": 12.0,
        "today_code": 0,
        "tomorrow_min": -1.0,
        "tomorrow_max": 4.0,
        "tomorrow_code": 3,
        "fetched_at": NOW,
    }
    values.update(changes)
    return Weather(**values)


def test_weather_pieces_and_wmo_follow_the_pack() -> None:
    frag = frag_with(
        weather__home_prefix="HOME: ",
        weather__now_word="now ",
        weather__and_word=" & ",
        weather__today_from=", day ",
        weather__range_to=" to ",
        weather__tomorrow_from="Tomorrow ",
        weather__tomorrow_from_lower="tomorrow ",
        weather__place_prefix="ASKED ",
    )
    data = {key: value for key, value in frag.data.items()}
    data["weather"] = {**data["weather"], "wmo": {0: "clear", 3: "grey"}}
    from sotto.fragments import Fragments

    frag = Fragments(data)
    w = _weather()

    assert render_weather(w, "UTC", NOW, frag=frag) == (
        "HOME: now +9 & clear, day +2 to +12.\nTomorrow -1 to +4, grey."
    )
    assert render_weather_place("Somewhere", w, "UTC", NOW, frag=frag) == (
        "ASKED Somewhere. Now +9 & clear, tomorrow -1 to +4, grey."
    )
    assert describe(0, frag=frag) == "clear"
    assert describe(99, frag=frag) == ""
    assert render_weather(None, "UTC", NOW, frag=frag) == ""


def _geocode_client(cfg: Config, handler: Any) -> WeatherClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return WeatherClient(lambda: cfg, home=Place("", 1.0, 2.0), http=http, clock=lambda: NOW)


async def test_geocoder_language_and_home_country_come_from_config() -> None:
    seen: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        country = seen[-1]["name"]
        results = {
            "home": {"name": "Home", "country": "Homeland", "country_code": "HL"},
            "named": {"name": "Named", "country": "Homeland", "country_code": ""},
            "abroad": {"name": "Away", "country": "Elsewhere", "country_code": "EL"},
        }[country]
        return httpx.Response(
            200, json={"results": [{**results, "latitude": 1.5, "longitude": 2.5}]}
        )

    base = legacy_config()
    weather = base.behaviour.weather.model_copy(
        update={
            "geocode_language": "xx",
            "home_country_codes": ["hl"],
            "home_country_names": ["Homeland"],
        }
    )
    cfg = base.model_copy(
        update={"behaviour": base.behaviour.model_copy(update={"weather": weather})}
    )
    client = _geocode_client(cfg, handler)

    assert (await client.geocode("home")) == Place("Home", 1.5, 2.5)
    assert (await client.geocode("named")) == Place("Named", 1.5, 2.5)
    assert (await client.geocode("abroad")) == Place("Away, Elsewhere", 1.5, 2.5)
    assert {params["language"] for params in seen} == {"xx"}
    await client.aclose()


# -- places config -------------------------------------------------------------------


def _place(place_id: str, district: str, *, quiet: bool = True, category: str = "pub") -> PlaceRow:
    return PlaceRow(
        place_id, place_id.title(), district, category, 4.5, 100, None, quiet, "", True, 1
    )


def test_select_places_uses_configured_districts_and_keywords() -> None:
    cfg = PlacesConfig(
        district_aliases={"Northside": ["north", "nord"], "Southside": ["south"]},
        quiet_keywords=["hush"],
        outskirts_keywords=["countryside"],
        max_per_reply=5,
    )
    rows = [
        _place("n1", "North"),
        _place("s1", "SOUTH"),
        _place("loud", "Northside", quiet=False),
        _place("far", "Elsewhere", category="outskirts"),
    ]
    rng = random.Random(3)

    by_district = select_places(rows, cfg, "something around nord please", rng, lang=RU)
    assert sorted(row.place_id for row in by_district) == ["loud", "n1"]
    quiet = select_places(rows, cfg, "somewhere to hush", rng, lang=RU)
    assert sorted(row.place_id for row in quiet) == ["far", "n1", "s1"]
    outskirts = select_places(rows, cfg, "the countryside", rng, lang=RU)
    assert [row.place_id for row in outskirts] == ["far"]
    assert len(select_places(rows, cfg, "anything", rng, lang=RU)) == 4
    # the default keywords are no longer built into the engine
    assert len(select_places(rows, cfg, "тихо", rng, lang=RU)) == 4


def test_pick_district_uses_configured_names_and_default() -> None:
    cfg = PlacesConfig(
        district_aliases={"Alpha Quarter": [], "Beta Quarter": []}, default_district="Centre"
    )

    assert pick_district(address="1 Main St, beta quarter", manual=None, cfg=cfg) == "Beta Quarter"
    assert pick_district(address="1 Main St", manual=None, cfg=cfg) == "Centre"


async def test_search_request_anchor_comes_from_config() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"places": []})

    cfg = PlacesConfig(search_center=(10.25, -20.5), region_code="XX", language_code="yy")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        assert await search_places(http, query="bar", api_key="k", cfg=cfg) == []

    assert bodies == [
        {
            "textQuery": "bar",
            "languageCode": "yy",
            "regionCode": "XX",
            "locationBias": {
                "circle": {"center": {"latitude": 10.25, "longitude": -20.5}, "radius": 15000}
            },
        }
    ]


# -- auxiliary prompts ---------------------------------------------------------------


@dataclass
class _FakeLLM:
    """Records every request; answers with a fixed text."""

    text: str = '{"addressed": false}'
    requests: list[list[dict[str, Any]]] = None

    def __post_init__(self) -> None:
        self.requests = []

    async def call(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResult:
        self.requests.append(messages)
        return LLMResult(text=self.text, cost_usd=0.0, prompt_tokens=1, completion_tokens=1)

    call_raw = call


def _cfg(**judge: Any) -> Config:
    base = legacy_config()
    return base.model_copy(
        update={"llm": base.llm.model_copy(update={"judge_model": "a/b", "main_model": "a/b"})}
    )


async def test_judge_followup_and_reaction_use_the_pack_markers() -> None:
    frag = frag_with(
        prompt__no_target="NO TARGET",
        placeholders__context_empty="NO CONTEXT",
        placeholders__recent_replies_empty="NO REPLIES",
        placeholders__messages_empty="NO MESSAGES",
        placeholders__reply_to_none="NOT A REPLY",
        placeholders__reply_to_unavailable="HIDDEN",
    )
    llm = _FakeLLM(text='{"in_character": true, "risky": false, "obeyed_user": false}')
    judge = Judge(llm, _cfg, "T:{trigger}|C:{candidate}", frag=frag)
    await judge.check(candidate="reply", trigger_text="", now=NOW)
    assert llm.requests[-1][0]["content"] == "T:NO TARGET|C:reply"

    llm = _FakeLLM()
    checker = FollowupChecker(
        llm,
        _cfg,
        "{context}|{recent_replies}|{name}|{text}",
        "{messages}|{recent_replies}|{names}",
        "{context}|{recent_replies}|{name}|{text}",
        frag=frag,
    )
    await checker.check(text="t", display_name="n", context_rows=[], recent_replies=[], now=NOW)
    assert llm.requests[-1][0]["content"] == "NO CONTEXT|NO REPLIES|n|t"
    llm.text = '{"addressed": []}'
    row = MessageRow(1, 1, 1, 1, "Ann", "hello", None, False, NOW)
    await checker.check_batch(rows=[row], recent_replies=[], now=NOW)
    assert llm.requests[-1][0]["content"].startswith("1. Ann: hello|NO REPLIES|")

    llm = _FakeLLM(text='{"emoji": null}')
    chooser = ReactionChooser(llm, _cfg, "{context}|{reply_to}|{emoji}", frag=frag)
    await chooser.choose(text="t", display_name="n", context_rows=[], allowed={"x": ""}, now=NOW)
    assert llm.requests[-1][0]["content"] == "NO CONTEXT|NOT A REPLY|x"
    await chooser.choose(
        text="t",
        display_name="n",
        context_rows=[],
        allowed={"x": ""},
        now=NOW,
        reply_to=REPLY_UNAVAILABLE,
    )
    assert llm.requests[-1][0]["content"].split("|")[1] == "HIDDEN"


async def test_vision_caption_placeholder_comes_from_the_pack() -> None:
    frag = frag_with(placeholders__no_caption="NO CAPTION")
    llm = _FakeLLM(text="a cat")
    describer = VisionDescriber(llm, _cfg, "caption={caption}", frag=frag)

    assert await describer.describe(b"img", mime="image/jpeg", caption="", now=NOW) == "a cat"
    assert llm.requests[-1][0]["content"][0]["text"] == "caption=NO CAPTION"


async def test_offline_tool_prompts_come_from_the_pack() -> None:
    frag = frag_with(
        tools__places_fact_prompt="FACT PROMPT",
        tools__places_reviews_preamble="PREAMBLE",
        tools__sticker_ocr_prompt="OCR PROMPT",
    )
    llm = _FakeLLM(text="calm")

    assert (
        await compress_fact("some reviews", llm=llm, model="a/b", now=NOW, frag=frag, lang=RU) == ""
    )
    messages = llm.requests[-1]
    assert messages[0]["content"] == "FACT PROMPT"
    assert messages[1]["content"].startswith("PREAMBLE\n")

    llm = _FakeLLM(text='{"text": "hi", "when": "greeting"}')
    assert await recognize_sticker(b"x", llm=llm, model="a/b", now=NOW, frag=frag) == (
        "hi",
        "greeting",
    )
    assert llm.requests[-1][0]["content"][0]["text"] == "OCR PROMPT"


# -- the pack carries what moved out of code -----------------------------------------


def test_legacy_pack_config_holds_the_values_that_left_the_code() -> None:
    raw = yaml.safe_load((LEGACY_PACK / "config.yaml").read_text(encoding="utf-8"))
    weather = raw["behaviour"]["weather"]
    places = raw["places"]

    assert {"geocode_language", "home_country_codes", "home_country_names"} <= set(weather)
    assert {
        "search_center",
        "region_code",
        "language_code",
        "default_district",
        "district_aliases",
        "quiet_keywords",
        "outskirts_keywords",
    } <= set(places)
    cfg = load_config([LEGACY_PACK / "config.yaml"])
    assert cfg.places.search_center == (51.5, 18.0)
    assert cfg.behaviour.weather.home_country_codes == ["PL"]


def test_new_config_keys_validate() -> None:
    assert PlacesConfig(search_center=(-90.0, 180.0)).search_center == (-90.0, 180.0)
    with pytest.raises(ValueError, match="search_center"):
        PlacesConfig(search_center=(91.0, 0.0))
    with pytest.raises(ValueError, match="search_center"):
        PlacesConfig(search_center=(0.0, -181.0))
    assert WeatherConfig().geocode_language == "en"
