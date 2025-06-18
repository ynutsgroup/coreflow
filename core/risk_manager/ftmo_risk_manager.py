#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FTMO Risk Manager – Institutional Grade

Features:
- Tagesverlustkontrolle
- Positionslimitüberwachung
- Verlustwarnung (Telegram)
- Optional erweiterbar durch EnhancedFTMOEngine
"""

import os
import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from aiogram import Bot
from aiogram.exceptions import TelegramAPIError

# Load .env
load_dotenv()

# Logger einrichten
logger = logging.getLogger("FTMO.RiskManager")
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler()]
    )

@dataclass
class FTMOConfig:
    TELEGRAM_TOKEN: str
    TELEGRAM_CHAT_ID: str
    MAX_DAILY_LOSS: float = 0.05
    MAX_OVERALL_LOSS: float = 0.10
    MAX_DAILY_TRADES: int = 5
    MAX_RISK_PER_TRADE: float = 0.02
    STARTING_BALANCE: float = 100000.0
    ALERT_THRESHOLD: float = 0.75

    def __post_init__(self):
        if not self.TELEGRAM_TOKEN or not self.TELEGRAM_CHAT_ID:
            raise ValueError("Telegram config fehlt")
        if self.STARTING_BALANCE <= 0:
            raise ValueError("STARTING_BALANCE muss > 0 sein")
        if not (0 < self.ALERT_THRESHOLD <= 1):
            raise ValueError("ALERT_THRESHOLD muss 0 < x <= 1 sein")

class FTMO_RiskManager:
    def __init__(self, config: Optional[FTMOConfig] = None):
        self.config = config or self._load_config()
        self.bot = Bot(token=self.config.TELEGRAM_TOKEN)
        self.daily_trade_count = 0
        self.daily_loss = 0.0
        self.overall_loss = 0.0
        self.last_reset_date = datetime.now(timezone.utc).date()
        logger.info("FTMO RiskManager ready")

    def _load_config(self) -> FTMOConfig:
        return FTMOConfig(
            TELEGRAM_TOKEN=os.getenv("TELEGRAM_TOKEN"),
            TELEGRAM_CHAT_ID=os.getenv("TELEGRAM_CHAT_ID"),
            MAX_DAILY_LOSS=float(os.getenv("MAX_DAILY_LOSS", "0.05")),
            MAX_OVERALL_LOSS=float(os.getenv("MAX_OVERALL_LOSS", "0.10")),
            MAX_DAILY_TRADES=int(os.getenv("MAX_DAILY_TRADES", "5")),
            MAX_RISK_PER_TRADE=float(os.getenv("MAX_RISK_PER_TRADE", "0.02")),
            STARTING_BALANCE=float(os.getenv("STARTING_BALANCE", "100000.0")),
            ALERT_THRESHOLD=float(os.getenv("ALERT_THRESHOLD", "0.75"))
        )

    def _auto_reset(self):
        today = datetime.now(timezone.utc).date()
        if self.last_reset_date != today:
            self.daily_trade_count = 0
            self.daily_loss = 0.0
            self.last_reset_date = today
            logger.info("🔄 Tageszähler zurückgesetzt")

    async def _send_telegram_alert(self, message: str):
        try:
            await self.bot.send_message(
                chat_id=self.config.TELEGRAM_CHAT_ID,
                text=f"🚨 *FTMO Alert*:\n{message}",
                parse_mode="Markdown"
            )
            logger.info("📢 Telegram-Alert gesendet")
        except TelegramAPIError as e:
            logger.error("Telegram API Error: %s", str(e))

    async def validate_trade(self, trade_value: float, risk_percentage: float) -> Tuple[bool, str]:
        self._auto_reset()

        if self.daily_trade_count >= self.config.MAX_DAILY_TRADES:
            msg = f"Max Trades erreicht: {self.config.MAX_DAILY_TRADES}"
            await self._send_telegram_alert(msg)
            return False, msg

        if abs(risk_percentage) > self.config.MAX_RISK_PER_TRADE:
            msg = f"Risiko {risk_percentage:.2%} zu hoch (Max: {self.config.MAX_RISK_PER_TRADE:.2%})"
            await self._send_telegram_alert(msg)
            return False, msg

        if trade_value < 0:
            projected_loss = self.daily_loss + abs(trade_value)
            limit = self.config.MAX_DAILY_LOSS * self.config.STARTING_BALANCE
            if projected_loss > limit:
                msg = f"Tagesverlust {projected_loss:.2f} > Limit {limit:.2f}"
                await self._send_telegram_alert(msg)
                return False, msg
            elif projected_loss > limit * self.config.ALERT_THRESHOLD:
                await self._send_telegram_alert(
                    f"⚠️ Verlust erreicht {projected_loss:.2f} ({projected_loss/limit:.0%} des Limits)"
                )

        self.daily_trade_count += 1
        if trade_value < 0:
            self.daily_loss += abs(trade_value)

        return True, "Trade validiert"

# === Erweiterung für KI-Engines ===
class EnhancedFTMOEngine(FTMO_RiskManager):
    """
    Erweiterung für VALGHybridEngine – erlaubt Plug&Play
    """
    def __init__(self, config: Optional[FTMOConfig] = None):
        super().__init__(config=config)
