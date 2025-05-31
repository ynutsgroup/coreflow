#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FTMO Compliance Manager v4.1 – Linux Version mit Telegram
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import requests
from typing import Optional, Dict, Tuple
from datetime import datetime, timezone

# .env laden
load_dotenv('/opt/coreflow/.env')

# Konfiguration aus .env
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "False").lower() == "true"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
LOG_DIR = os.getenv("LOG_DIR", "/opt/coreflow/logs")

# FTMO Compliance Regeln
FTMO_RULES = {
    'MAX_DAILY_LOSS': float(os.getenv("DAILY_LOSS_LIMIT", "5.0")) / 100,  # z. B. 5%
    'MAX_OVERALL_LOSS': 0.10,
    'MAX_RISK_PER_TRADE': float(os.getenv("MAX_RISK_PERCENT", "1.0")) / 100,
    'MAX_DAILY_TRADES': 5,
    'MIN_RISK_PER_TRADE': 0.005
}


class FTMOComplianceManager:
    def __init__(self, starting_balance: float):
        self.starting_balance = starting_balance
        self.current_balance = starting_balance
        self.daily_stats = {
            'trades': 0,
            'loss': 0.0,
            'profit': 0.0,
            'last_reset': datetime.now(timezone.utc).date()
        }
        self._setup_logging()
        self._validate_rules()

    def _setup_logging(self):
        log_path = os.path.join(LOG_DIR, "ftmo/compliance.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        self.logger = logging.getLogger("FTMOCompliance")
        self.logger.setLevel(logging.INFO)

        handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _validate_rules(self):
        if self.starting_balance <= 0:
            raise ValueError("Startbalance muss positiv sein")

    def _send_telegram_alert(self, message: str):
        if not TELEGRAM_ENABLED:
            return
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML"
            }
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            self.logger.warning(f"Telegram send failed: {e}")

    def _check_daily_reset(self):
        today = datetime.now(timezone.utc).date()
        if self.daily_stats['last_reset'] != today:
            self.daily_stats = {
                'trades': 0,
                'loss': 0.0,
                'profit': 0.0,
                'last_reset': today
            }
            self.logger.info("Tägliches Reset durchgeführt")

    def validate_trade(self, risk_percent: float) -> Tuple[bool, str]:
        self._check_daily_reset()

        if self.daily_stats['trades'] >= FTMO_RULES['MAX_DAILY_TRADES']:
            return False, f"Maximale Anzahl an Trades pro Tag ({FTMO_RULES['MAX_DAILY_TRADES']}) erreicht."

        if not FTMO_RULES['MIN_RISK_PER_TRADE'] <= risk_percent <= FTMO_RULES['MAX_RISK_PER_TRADE']:
            return False, f"Risikoprozentsatz {risk_percent:.2%} liegt außerhalb der erlaubten Range."

        daily_loss_limit = self.starting_balance * FTMO_RULES['MAX_DAILY_LOSS']
        if self.daily_stats['loss'] >= daily_loss_limit:
            return False, f"Tägliches Verlustlimit ({daily_loss_limit:.2f}) erreicht."

        return True, "Trade genehmigt."

    def register_trade(self, pnl: float, risk_percent: float) -> bool:
        is_valid, reason = self.validate_trade(risk_percent)
        if not is_valid:
            self.logger.warning(f"Trade abgelehnt: {reason}")
            self._send_telegram_alert(f"🚫 <b>Trade abgelehnt</b>\nGrund: {reason}")
            return False

        self.daily_stats['trades'] += 1
        if pnl < 0:
            self.daily_stats['loss'] += abs(pnl)
        else:
            self.daily_stats['profit'] += pnl

        self.current_balance += pnl

        self.logger.info(f"Trade registriert: PnL {pnl:.2f} | Risiko {risk_percent:.2%}")
        return True

    def check_equity_protection(self) -> bool:
        overall_loss = self.starting_balance - self.current_balance
        max_loss = self.starting_balance * FTMO_RULES['MAX_OVERALL_LOSS']
        if overall_loss >= max_loss:
            msg = f"❗ <b>Equity Protection aktiviert</b>\nVerlust {overall_loss:.2f} überschreitet Limit {max_loss:.2f}"
            self.logger.critical(msg)
            self._send_telegram_alert(msg)
            return False
        return True


if __name__ == "__main__":
    manager = FTMOComplianceManager(starting_balance=100000)

    # Beispiel-Trades
    trades = [
        (1500, 0.01),
        (-3000, 0.015),
        (2000, 0.01),
        (-4000, 0.02),
        (1000, 0.005),
        (-500, 0.01)
    ]

    for pnl, risk in trades:
        result = manager.register_trade(pnl, risk)
        print("✅" if result else "❌", pnl, f"{risk:.2%}")

    if not manager.check_equity_protection():
        print("⚠️ Equity Limit überschritten!")
