"""The owner alert rendered from the legacy Russian pack's locale (a pack locale overrides the
engine's English catalog).

Migration proof: removed together with ``personas/_legacy_ru`` in PR-9.
"""

from __future__ import annotations

import asyncio

from sotto.alerts import OwnerAlerter
from sotto.config_models import Config
from sotto.persona import Persona
from tests.test_alerts import ADMIN_ID, NOW, FakeBot, FakeStore


async def test_alert_text_comes_from_the_persona_locale(legacy_pack: Persona) -> None:
    bot, store = FakeBot(), FakeStore()
    alerter = OwnerAlerter(
        bot=bot,
        db=store,
        admin_user_id=ADMIN_ID,
        cfg_getter=Config,
        clock=lambda: NOW,
        persona=legacy_pack,
    )
    await alerter.provider_refused(status=402, model="m", message="no credits")
    await asyncio.sleep(0.05)
    assert "кончились кредиты" in bot.sent[0][1]
