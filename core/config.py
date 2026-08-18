"""Runtime configuration, loaded from environment / .env.

MASTER_PLAN §13.7: credentials are separated by purpose. Research code reads
`Settings`; broker credentials live in a distinct object that the research and
AI layers have no import path to (§3.2).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Environment", "Settings", "settings"]


class Environment(str, Enum):
    DEV = "dev"
    PAPER = "paper"
    LIVE = "live"

    @property
    def is_live(self) -> bool:
        return self is Environment.LIVE


class Settings(BaseSettings):
    """Non-secret configuration.

    Deliberately excludes broker credentials: nothing that can place an order
    is reachable from a research process.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="NEUTRON_",
        extra="ignore",
        frozen=True,
    )

    env: Environment = Environment.DEV
    lake_path: Path = Path("./lake")

    # 127.0.0.1 rather than localhost: on Windows localhost resolves to ::1
    # first while Docker publishes IPv4 only, turning a failure into a hang.
    db_url: str = "postgresql+psycopg://neutron:neutron@127.0.0.1:5433/neutron"

    alpaca_data_url: str = "https://data.alpaca.markets/v2"
    alpaca_paper_url: str = "https://paper-api.alpaca.markets/v2"

    #: Default-off. No real order may leave the process unless this is
    #: explicitly true, whatever any strategy or agent believes (§21).
    live_enabled: bool = Field(default=False)

    #: Telegram is the primary alarm, not the console and not the UI (§M9).
    #: Empty means alerts still fire — to the console — but nothing will wake
    #: you, and the M9 gate explicitly requires having been woken once.
    #: These are chat-routing identifiers rather than trading credentials, so
    #: they live here and not in `core.secrets`; they cannot place an order.
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    @property
    def telegram_configured(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    @property
    def lake(self) -> Path:
        path = self.lake_path.expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def require_live_permission(self) -> None:
        """Guard every real-money path.

        Raises:
            PermissionError: unless the environment is LIVE *and* live trading
                was explicitly enabled. Both, not either.
        """
        if not (self.env.is_live and self.live_enabled):
            raise PermissionError(
                f"live trading blocked: env={self.env.value}, "
                f"live_enabled={self.live_enabled}. Both must be set (§21)."
            )


settings = Settings()
