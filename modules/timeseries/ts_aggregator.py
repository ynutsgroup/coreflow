#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoreFlow TS Aggregator v2.5 – Institutional AI/FTMO Edition
📉 Redis TimeSeries Aggregation | 🔐 .env.enc Support | ✅ Audit-Log | ✅ Drawdown-Wächter
✅ AI-Fähig (RSI, VWAP, SMA) | ✅ Pandas-Bridge | 🔒 Dezentral konfigurierbar
"""

import os
import sys
import logging
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from concurrent.futures import ThreadPoolExecutor

# === 🔐 Dynamischer Import des verschlüsselten .env Loaders ===
decrypt_path = Path("/opt/coreflow/utils")
if decrypt_path.exists():
    sys.path.insert(0, str(decrypt_path))
else:
    raise FileNotFoundError(f"❌ decrypt_env.py nicht gefunden: {decrypt_path}")

from decrypt_env import load_env, find_latest_env_enc

env_path = find_latest_env_enc()
if not env_path:
    raise FileNotFoundError("❌ Keine gültige .env.enc-Datei gefunden.")
env = load_env(env_path, "/opt/coreflow/infra/vault/encryption.key")

# === Redis-Client vorbereiten ===
from redis.commands.ts.client import TsClient
from redis import Redis

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6380"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

redis_raw = Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True)
ts_client = TsClient(redis_raw)

# === 🔐 FTMO Audit Logging ===
class FTMO_Logger:
    def __init__(self):
        self.logger = logging.getLogger("ftmo_audit")
        self.logger.setLevel(logging.INFO)

        log_path = os.getenv("FTMO_AUDIT_LOG", "/opt/coreflow/logs/ftmo_audit.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s|%(levelname)s|%(message)s|HASH:%(hash)s'
        ))
        self.logger.addHandler(handler)

    def log(self, message: str):
        log_hash = hashlib.sha256(message.encode()).hexdigest()[:16]
        self.logger.info(message, extra={'hash': log_hash})

# === 📉 Drawdown-Monitor (FTMO-Konformität) ===
class RiskEngine:
    def __init__(self, max_drawdown=5.0):
        self.max_dd = max_drawdown
        self.equity_high = 0.0

    def update(self, current_equity: float) -> bool:
        self.equity_high = max(self.equity_high, current_equity)
        drawdown = (self.equity_high - current_equity) / self.equity_high * 100
        return drawdown >= self.max_dd

# === ⏱️ Redis OHLCV Aggregation mit Pandas ===
def _redis_ohlcv_aggregation(symbol: str, tf: str, from_ts: int, to_ts: int) -> pd.DataFrame:
    logger = logging.getLogger("coreflow.ts_aggregator")
    logger.info(f"🧠 Redis Aggregation: TS.MRANGE {from_ts} → {to_ts} | symbol={symbol.upper()}")

    try:
        res = ts_client.mrange(
            from_ts,
            to_ts,
            filters=[f"symbol={symbol.lower()}"],
            with_labels=True
        )

        if not res:
            raise ValueError("❌ Keine Zeitreihendaten gefunden.")

        data = []
        for stream in res:
            for ts, val in stream['data']:
                dt = datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc)
                data.append((dt, float(val)))

        df = pd.DataFrame(data, columns=["datetime", "value"])
        df.set_index("datetime", inplace=True)
        return df

    except Exception as e:
        logger.error(f"Redis aggregation failed: {e}", exc_info=True)
        raise

# === 🧠 AI Feature Engineering ===
def add_ai_features(df: pd.DataFrame) -> pd.DataFrame:
    df['vwap'] = (df['value'] * np.arange(1, len(df) + 1)).cumsum() / np.arange(1, len(df) + 1)
    df['sma20'] = df['value'].rolling(20).mean()
    delta = df['value'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

# === 📈 Aggregierter Datensatz mit AI Features liefern ===
def get_ohlcv(symbol: str, timeframe: str = '1m') -> pd.DataFrame:
    now = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    from_ts = now - 3_600_000  # letzte 1 Stunde

    df = _redis_ohlcv_aggregation(symbol, timeframe, from_ts, now)
    if df.empty:
        raise ValueError("❌ Keine Daten zurückgegeben.")
    return add_ai_features(df)

# === 📤 Mocked Order-Platzierung (für reale MT5-Integration anpassbar) ===
def place_order(order: str):
    print(f"📤 Order platziert: {order}")

# === 🧠 AI/FTMO Strategy Execution Loop ===
def execute_trade(symbol: str, strategy: Callable[[pd.DataFrame], bool]):
    ftmo_log = FTMO_Logger()
    risk = RiskEngine(max_drawdown=5.0)

    while True:
        try:
            df = get_ohlcv(symbol)
            signal = strategy(df)

            current_equity = 100000.0  # In produktiver Version via Broker abrufen

            if signal and not risk.update(current_equity):
                order = f"BUY {symbol} @ {df['value'].iloc[-1]:.5f}"
                ftmo_log.log(order)
                place_order(order)

        except Exception as e:
            ftmo_log.log(f"ERROR: {str(e)}")
            break  # oder: time.sleep(60)

# === 🧪 Beispielstrategie (Mean-Reversion) ===
def mean_reversion_strategy(df: pd.DataFrame) -> bool:
    return df['value'].iloc[-1] < df['sma20'].iloc[-1] * 0.98

# === 🚀 MAIN ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    execute_trade("EURUSD", mean_reversion_strategy)
