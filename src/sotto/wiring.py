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
    # Отказ провайдера (кончились кредиты, ключ не принят) — сообщение владельцу
    # в личку (CLAUDE.md, "Интерфейсы: тревога владельцу").
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

    # Дешёвая проверка «это мне?» в горячем окне (CLAUDE.md, "внимание как
    # у живого человека") — создаётся вместе с остальными LLM-зависимыми
    # компонентами, как только есть ключ; enabled/model проверяются на
    # каждом вызове, а не один раз при старте.
    followup_prompt = persona.prompts["followup"]
    # Дешёвый предфильтр "вернулся проверить" (CLAUDE.md, "Интерфейсы:
    # дешёвый предфильтр для «вернулся проверить»") — тот же чекер, второй
    # промпт: пачка сообщений вместо одного, перед основной моделью.
    checkin_prefilter_prompt = persona.prompts["checkin_prefilter"]
    # Подтверждение обращения по имени (CLAUDE.md, "имя в падежах и
    # проверка обращения по имени") — тот же чекер, третий промпт: имя
    # в любом падеже нашла регулярка, а обращение это или разговор о
    # нём в третьем лице, решает дешёвая модель.
    name_check_prompt = persona.prompts["name_check"]
    followup_checker = FollowupChecker(
        llm,
        config_store.get,
        followup_prompt,
        checkin_prefilter_prompt,
        name_check_prompt,
        frag=persona.frag,
    )

    # Зрение на фото (CLAUDE.md, "зрение на фото") — как и followup,
    # создаётся вместе с остальными LLM-зависимыми компонентами; enabled и
    # модель проверяются на каждом снимке, а не один раз при старте.
    vision_prompt = persona.prompts["vision"]
    vision = VisionDescriber(llm, config_store.get, vision_prompt, frag=persona.frag)

    # Выбор реакции моделью (CLAUDE.md, "реакции с задержкой и смыслом") —
    # как followup и зрение, создаётся при наличии ключа; модель и
    # semantic проверяются на каждом вызове, а не один раз при старте.
    reaction_prompt = persona.prompts["reaction"]
    reaction_chooser = ReactionChooser(llm, config_store.get, reaction_prompt, frag=persona.frag)

    # Каталог стикеров — офлайн-файл (CLAUDE.md, "Интерфейсы: стикеры"),
    # правится stickers_fill.py и владельцем руками. Пустой/отсутствующий
    # файл -> пустой каталог, чузер не создаётся, всё остальное работает
    # как раньше (Responder.sticker_chooser=None).
    catalog = load_catalog(persona.stickers_path)
    sticker_catalog_enabled = sum(1 for sticker in catalog.stickers if sticker.enabled)
    sticker_chooser: StickerChooser | None = None
    if sticker_catalog_enabled:
        sticker_prompt = persona.prompts["sticker"]
        sticker_chooser = StickerChooser(
            llm, config_store.get, catalog, sticker_prompt, frag=persona.frag
        )

    # Долгая память чата (CLAUDE.md, "долгая память чата"): фоновый таск
    # раз в час смотрит, попало ли локальное время в run_window, и
    # пересказывает завершённые периоды. enabled проверяется на каждом
    # заходе, а не один раз при старте — /set включает без рестарта.
    memory_prompt = persona.prompts["memory"]
    # Два дополнительных вызова недельного пересказа: шутки чата и незакрытые
    # истории людей (CLAUDE.md, "шутки чата и истории людей").
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

    # Место из вопроса про погоду достаёт дешёвая модель (CLAUDE.md,
    # "погода в другом месте") — как followup и зрение, создаётся при
    # наличии ключа; модель и place_lookup проверяются на каждом вызове.
    weather_place_prompt = persona.prompts["weather_place"]
    weather_places = WeatherPlaceExtractor(llm, config_store.get, weather_place_prompt)

    # Дневник дня (CLAUDE.md, "дневник дня"): дешёвое извлечение факта о себе
    # из каждой своей текстовой реплики; модель и enabled — на каждом вызове.
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
        # Та же дешёвая проверка "это мне?", что и в горячем окне: Responder
        # перепроверяет ею строку, которую выбрал checkin (CLAUDE.md, "меньше
        # и разнообразнее", мера 3).
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
