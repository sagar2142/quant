"""Alert routing from configuration — MASTER_PLAN §M9, §12.7.

One place that decides which channels an alert reaches, so no caller has to
remember. `ops.alerts` knows *how* to send; this knows *where*.

**The console sink is always present.** A misconfigured Telegram token must
degrade to a printed alert, never to silence — an alert that vanishes because
of a typo in an environment variable is worse than no alerting system, because
you believe you have one.

**Telegram is the alarm; the console is the record.** §M9 gates live trading on
having been woken by an alert at least once, and a line in a log file has never
woken anybody. When the token is absent the router says so out loud at
construction, rather than at 3am when the position is wrong.
"""

from __future__ import annotations

import logging

from core.config import Settings, settings
from ops.alerts import AlertRouter, AlertSink, ConsoleSink, TelegramSink

__all__ = ["build_router", "describe_channels"]

logger = logging.getLogger(__name__)


def build_router(config: Settings | None = None) -> AlertRouter:
    """Every configured channel, console always included.

    Args:
        config: Injected for testing. Defaults to the process settings.
    """
    active = config or settings
    sinks: list[AlertSink] = [ConsoleSink()]

    if active.telegram_configured:
        sinks.append(
            TelegramSink(
                bot_token=active.telegram_bot_token,
                chat_id=active.telegram_chat_id,
            )
        )
    else:
        logger.warning(
            "Telegram is not configured — alerts will print and nothing will "
            "wake you. Set NEUTRON_TELEGRAM_BOT_TOKEN and "
            "NEUTRON_TELEGRAM_CHAT_ID (§M9)."
        )
    return AlertRouter(sinks)


def describe_channels(config: Settings | None = None) -> str:
    """One line naming the live channels, for a CLI to print at startup.

    Printed rather than assumed: the operator should know before a bad session
    whether anything is going to reach them during one.
    """
    active = config or settings
    if active.telegram_configured:
        return "alerts: console + telegram"
    return "alerts: console only — nothing will wake you (§M9)"
