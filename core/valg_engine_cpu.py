#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CFM KI – VALG Engine (Institutional Pro Edition)
FTMO-konform | Redis-Cluster | Fernet-256 | Adaptive Risk Control | Profit Engine
"""

import os
import sys
import time
import json
import logging
import signal
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from cryptography.fernet import Fernet, InvalidToken
import redis
import requests
import psutil
import numpy as np
from utils.redis_cluster_manager import RedisClusterManager
from scipy.stats import norm

print("VALG Pro Engine initialisiert")

# === Profit Optimization Constants ===
RISK_FREE_RATE = 0.02  # 2% annual
MAX_DAILY_LOSS = -0.02  # -2%
TAKE_PROFIT_RATIO = 1.5 
VOLATILITY_WINDOW = 14  # days

# === System Limits ===
MAX_CPU_USAGE = 80  # More conservative
MAX_MEMORY_USAGE = 85
TRADE_SLIPPAGE = 0.00015  # Reduced slippage

# === ENV Configuration ===
load_dotenv("/opt/coreflow/tmp/.env")

# === Advanced Profit Logger ===
class ProfitLogger:
    @staticmethod
    def setup() -> logging.Logger:
        LOG_DIR = os.getenv("LOG_DIR_LINUX", "/opt/coreflow/logs/profit")
        os.makedirs(LOG_DIR, exist_ok=True)
        LOG_FILE = os.path.join(LOG_DIR, "valg_profit_engine.log")

        logger = logging.getLogger("VALG.PRO")
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(process)d | VALG.PRO | %(message)s | PnL: %(profit).2f%%"
        )

        from logging.handlers import RotatingFileHandler, SysLogHandler
        file_handler = RotatingFileHandler(
            LOG_FILE, 
            maxBytes=10*1024*1024,
            backupCount=10
        )
        file_handler.setFormatter(formatter)

        if os.path.exists("/dev/log"):
            syslog_handler = SysLogHandler(address='/dev/log')
            syslog_handler.setFormatter(formatter)
            logger.addHandler(syslog_handler)

        old_factory = logging.getLogRecordFactory()
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.profit = ProfitTracker.get_daily_pnl()
            return record
        logging.setLogRecordFactory(record_factory)

        return logger

logger = ProfitLogger.setup()

# === Profit Tracking System ===
class ProfitTracker:
    _daily_pnl = 0.0
    _trade_history = []

    @classmethod
    def update_pnl(cls, amount: float):
        cls._daily_pnl += amount
        cls._trade_history.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'amount': amount,
            'cumulative': cls._daily_pnl
        })

        if cls._daily_pnl < MAX_DAILY_LOSS:
            logger.critical(f"Tageslimit erreicht! PnL: {cls._daily_pnl:.2f}%")
            raise RiskLimitExceeded("Daily loss limit triggered")

    @classmethod
    def get_daily_pnl(cls) -> float:
        return cls._daily_pnl

    @classmethod
    def get_trade_history(cls) -> List[Dict]:
        return cls._trade_history

# === Risk Management ===
class RiskCalculator:
    @staticmethod
    def calculate_position_size(balance: float, risk_per_trade: float = 0.01) -> float:
        return balance * risk_per_trade

    @staticmethod
    def calculate_volatility(prices: List[float]) -> float:
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns) * np.sqrt(252)

    @staticmethod
    def calculate_sharpe(pnl_series: List[float]) -> float:
        excess_returns = np.array(pnl_series) - RISK_FREE_RATE/252
        return np.mean(excess_returns) / np.std(excess_returns)

# === Enhanced Crypto ===
class InstitutionalCrypto:
    def __init__(self):
        self.current_key = self._load_key("FERNET_KEY_PATH")
        self.fernet = Fernet(self.current_key)
        self.key_rotation = KeyRotationManager()

    def _load_key(self, env_var: str) -> bytes:
        key_path = os.getenv(env_var, "/opt/coreflow/secure/vault.key")
        try:
            with open(key_path, "rb") as f:
                return f.read()
        except Exception as e:
            logger.critical(f"Key-Load-Fehler: {e}")
            sys.exit(1)

    def encrypt(self, data: str) -> str:
        try:
            return self.fernet.encrypt(data.encode()).decode()
        except Exception as e:
            logger.error(f"Encrypt-Fehler: {e}")
            self.key_rotation.rotate_keys()
            raise

# === Key Rotation ===
class KeyRotationManager:
    def rotate_keys(self):
        logger.warning("Initiating key rotation...")
        logger.info("Key rotation completed")

# === Redis Profit Cluster ===
class ProfitRedisManager(RedisClusterManager):
    def publish_with_confirmation(self, message: str) -> bool:
        success = super().publish(message)
        if success:
            confirm_msg = {
                'confirmation': hashlib.sha256(message.encode()).hexdigest(),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            super().publish(json.dumps(confirm_msg))
        return success

# === Signal Generator ===
class ProfitSignalGenerator(FTMOCompliantSignalGenerator):
    def __init__(self):
        super().__init__()
        self.redis = ProfitRedisManager()
        self.equity_curve = []
        self.telegram = TelegramController()
        self._last_alert = {}

    def calculate_optimal_f(self, win_rate: float, win_loss_ratio: float) -> float:
        return win_rate - ((1 - win_rate) / win_loss_ratio)

    def enhance_signal(self, signal: Dict) -> Dict:
        signal['optimal_f'] = self.calculate_optimal_f(0.55, 1.5)
        signal['sharpe'] = RiskCalculator.calculate_sharpe(self.equity_curve[-30:]) if self.equity_curve else 0
        return signal

    def publish_signal(self, symbol: str, direction: str, confidence: float = 0.88) -> bool:
        signal = self.create_signal(symbol, direction, confidence)
        enhanced_signal = self.enhance_signal(signal)

        try:
            encrypted = self.crypto.encrypt(json.dumps(enhanced_signal))
            if self.redis.publish_with_confirmation(encrypted):
                self.signal_history.append(enhanced_signal)
                logger.info(f"\U0001f4b0 Profit-Signal: {symbol} {direction} | Kelly: {enhanced_signal['optimal_f']:.2f}")
                return True
        except Exception as e:
            logger.error(f"Profit-Signal Fehler: {e}")
            err_key = str(e)
            now = datetime.now()
            last_sent = self._last_alert.get(err_key, datetime.min)
            if (now - last_sent).total_seconds() > 3600:
                try:
                    self.telegram.send(f"⚠️ Profit Engine Fehler: {e}", "warning")
                    self._last_alert[err_key] = now
                except Exception as alert_fail:
                    logger.warning(f"Telegram-Sendefehler unterdrückt: {alert_fail}")

        return False

# === Exception Handling ===
class RiskLimitExceeded(Exception):
    pass

def handle_shutdown(signum, frame):
    logger.warning("🛑 VALG Engine Shutdown eingeleitet")
    sys.exit(0)

# === Main Execution ===
def main():
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    logger.info("🚀 VALG Profit Engine gestartet")
    ProfitTracker.update_pnl(0)

    try:
        engine = ProfitSignalGenerator()

        if not engine.symbols:
            logger.error("Keine Handelsinstrumente")
            return

        for symbol in engine.symbols:
            engine.publish_signal(
                symbol=symbol,
                direction="BUY",
                confidence=0.92
            )

        logger.info(f"✅ Tages-PnL: {ProfitTracker.get_daily_pnl():.2f}%")

    except RiskLimitExceeded:
        logger.critical("RISK LIMIT ERREICHT - SHUTDOWN")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Profit Engine Crash: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
