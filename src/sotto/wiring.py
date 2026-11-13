"""Wiring of the LLM-dependent components, shared by ``app.main`` and ``packcheck --boot``.

One function builds everything that needs an API key (client, judge, checkers, describers,
memorizer, responder), so the boot dry-run exercises exactly the construction the process does.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import httpx

from sotto.alerts import OwnerAlerter
from sotto.chat_memory import ChatMemorizer
from sotto.db import Database
from sotto.diary import DiaryExtractor
from sotto.followup import FollowupChecker
from sotto.judge import Judge
from sotto.llm import LLMClient
from sotto.persona import Persona
from sotto.reactions import ReactionChooser
from sotto.responder import Responder, _BotLike
from sotto.stickers import StickerChooser, load_catalog
from sotto.stores import ConfigStore, PromptStore
from sotto.vision import VisionDescriber
from sotto.vocab import VocabStore
from sotto.weather import WeatherClient, WeatherPlaceExtractor


@dataclass
class LlmStack:
    llm: LLMClient
    alerter: OwnerAlerter
    judge: Judge
    followup: FollowupChecker
    vision: VisionDescriber
    reaction_chooser: ReactionChooser
    sticker_chooser: StickerChooser | None
    sticker_catalog_enabled: int
    memorizer: ChatMemorizer
    weather_places: WeatherPlaceExtractor
    diary: DiaryExtractor
    responder: Responder


def build_llm_stack(
    *,
    persona: Persona,
    config_store: ConfigStore,
    prompt_store: PromptStore,
    db: Database,
    bot: _BotLike,
    weather: WeatherClient,
    vocab_store: VocabStore,
    rng: random.Random,
    api_key: str,
    http: httpx.AsyncClient,
    admin_user_id: int,
    chat_id: int,
    bot_user_id: int,
) -> LlmStack:
    # Provider refusal (credits ran out, key rejected): message the owner in a DM.
    alerter = OwnerAlerter(
        bot=bot,
        db=db,
        admin_user_id=admin_user_id,
        cfg_getter=config_store.get,
        persona=persona,
    )
    llm = LLMClient(
        api_key=api_key,
        cfg_getter=config_store.get,
        db=db,
        http=http,
        on_refusal=alerter.provider_refused,
    )

    judge = Judge(llm, config_store.get, persona.prompts["judge"], frag=persona.frag)

    # Cheap "is this for me?" check in the hot window; created with the other
    # LLM-dependent components as soon as there is a key; enabled/model are checked
    # on every call, not once at startup.
    followup_prompt = persona.prompts["followup"]
    # Cheap prefilter for the "came back to check" pass: same checker, second
    # prompt: a batch of messages instead of one, before the main model.
    checkin_prefilter_prompt = persona.prompts["checkin_prefilter"]
    # Confirmation of an address by name: same checker, third prompt. A regex finds the
    # name in any case form; whether it is an address or talk about the bot in the
    # third person is decided by the cheap model.
    name_check_prompt = persona.prompts["name_check"]
    followup_checker = FollowupChecker(
        llm,
        config_store.get,
        followup_prompt,
        checkin_prefilter_prompt,
        name_check_prompt,
        frag=persona.frag,
    )

    # Photo vision: like followup, created with the other LLM-dependent components;
    # enabled and model are checked on every photo, not once at startup.
    vision_prompt = persona.prompts["vision"]
    vision = VisionDescriber(llm, config_store.get, vision_prompt, frag=persona.frag)

    # Model-based reaction choice: like followup and vision, created when there is a
    # key; model and semantic are checked on every call, not once at startup.
    reaction_prompt = persona.prompts["reaction"]
    reaction_chooser = ReactionChooser(llm, config_store.get, reaction_prompt, frag=persona.frag)

    # Sticker catalog is an offline file, edited by stickers_fill.py and by the owner
    # by hand. An empty/missing file -> empty catalog, no chooser is created, everything
    # else works as before (Responder.sticker_chooser=None).
    catalog = load_catalog(persona.stickers_path)
    sticker_catalog_enabled = sum(1 for sticker in catalog.stickers if sticker.enabled)
    sticker_chooser: StickerChooser | None = None
    if sticker_catalog_enabled:
        sticker_prompt = persona.prompts["sticker"]
        sticker_chooser = StickerChooser(
            llm, config_store.get, catalog, sticker_prompt, frag=persona.frag
        )

    # Long-term chat memory: a background task checks once an hour whether local time
    # is inside run_window and summarizes completed periods. enabled is checked on
    # every pass, not once at startup, so /set enables it without a restart.
    memory_prompt = persona.prompts["memory"]
    # Two extra calls of the weekly summary: chat jokes and unfinished stories of people.
    jokes_prompt = persona.prompts["memory_jokes"]
    threads_prompt = persona.prompts["memory_threads"]
    memorizer = ChatMemorizer(
        llm,
        db,
        config_store.get,
        memory_prompt,
        chat_id=chat_id,
        jokes_prompt=jokes_prompt,
        threads_prompt=threads_prompt,
        patterns_getter=config_store.patterns,
    )

    # The place from a weather question is extracted by the cheap model, like followup
    # and vision; created when there is a key; model and place_lookup are checked per call.
    weather_place_prompt = persona.prompts["weather_place"]
    weather_places = WeatherPlaceExtractor(llm, config_store.get, weather_place_prompt)

    # Daily diary: cheap extraction of a fact about itself from each own text reply;
    # model and enabled are checked on every call.
    diary_prompt = persona.prompts["diary"]
    diary = DiaryExtractor(llm, config_store.get, diary_prompt)

    responder = Responder(
        bot=bot,
        db=db,
        cfg_getter=config_store.get,
        llm=llm,
        judge=judge,
        sticker_chooser=sticker_chooser,
        weather=weather,
        weather_places=weather_places,
        diary=diary,
        vocab_getter=vocab_store.get,
        patterns_getter=config_store.patterns,
        prompt_store=prompt_store,
        rng=rng,
        chat_id=chat_id,
        bot_user_id=bot_user_id,
        # The same cheap "is this for me?" check as in the hot window: Responder
        # rechecks with it the line chosen by checkin.
        followup=followup_checker,
        persona=persona,
        frag=persona.frag,
    )

    return LlmStack(
        llm=llm,
        alerter=alerter,
        judge=judge,
        followup=followup_checker,
        vision=vision,
        reaction_chooser=reaction_chooser,
        sticker_chooser=sticker_chooser,
        sticker_catalog_enabled=sticker_catalog_enabled,
        memorizer=memorizer,
        weather_places=weather_places,
        diary=diary,
        responder=responder,
    )
