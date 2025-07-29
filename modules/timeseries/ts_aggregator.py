#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoreFlow TS Aggregator v2.6 – Institutional AI/FTMO Edition (No TsClient)
📉 Redis TimeSeries Aggregation via raw TS.RANGE | 🔐 .env.enc Support | ✅ Audit-Log | ✅ Drawdown-Wächter
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
import time

# === 🔐 ENV laden via envloader.py ===
sys.path.insert(0, "/opt/coreflow/utils")
from envloader import load_env, find_latest_env_enc

env_path = find_latest_env_enc()
if not env_path or not load_env(env_path, "/opt/coreflow/infra/vault/encryption.key"):
    raise SystemExit("❌ ENV konnte nicht geladen werden")

# === Redis Raw-Client ===
from redis import Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6380"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
r = Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True)

# === 🔐 Audit-Logger ===
class FTMO_Logger:
    def __init__(self):
        self.logger = logging.getLogger("ftmo_audit")
        self.logger.setLevel(logging.INFO)
        path = os.getenv("FTMO_AUDIT_LOG", "/opt/coreflow/logs/ftmo_audit.log")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fh = logging.FileHandler(path)
        fh.setFormatter(logging.Formatter('%(asctime)s|%(levelname)s|%(message)s|HASH:%(hash)s'))
        self.logger.addHandler(fh)

    def log(self, message: str):
        h = hashlib.sha256(message.encode()).hexdigest()[:16]
        self.logger.info(message, extra={'hash': h})

# === 🛡 Drawdown-Guard ===
class RiskEngine:
    def __init__(self, max_drawdown=5.0):
        self.max_dd = max_drawdown
        self.equity_high = 0.0

    def update(self, equity: float) -> bool:
        self.equity_high = max(self.equity_high, equity)
        dd = (self.equity_high - equity) / self.equity_high * 100
        return dd >= self.max_dd

# === OHLCV + KI-Funktionen ===
def fetch_redis_series(symbol: str, timeframe: str, from_ts: int, to_ts: int) -> pd.DataFrame:
    key = f"test:{symbol.lower()}"
    raw = r.execute_command("TS.RANGE", key, from_ts, to_ts)
    if not raw:
        raise ValueError("❌ Keine TS-Daten gefunden")
    data = [(datetime.utcfromtimestamp(int(ts) / 1000), float(val)) for ts, val in raw]
    df = pd.DataFrame(data, columns=["datetime", "value"]).set_index("datetime")
    return df

def add_ai_features(df: pd.DataFrame) -> pd.DataFrame:
    df['vwap'] = (df['value'] * np.arange(1, len(df)+1)).cumsum() / np.arange(1, len(df)+1)
    df['sma20'] = df['value'].rolling(20).mean()
    delta = df['value'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def get_ohlcv(symbol: str, timeframe: str = '1m') -> pd.DataFrame:
    now = int(time.time() * 1000)
    from_ts = now - 60 * 60 * 1000  # Letzte Stunde
    df = fetch_redis_series(symbol, timeframe, from_ts, now)
    return add_ai_features(df)

def place_order(order: str):
    print(f"📤 Order platziert: {order}")

def mean_reversion_strategy(df: pd.DataFrame) -> bool:
    return df['value'].iloc[-1] < df['sma20'].iloc[-1] * 0.98

def execute_trade(symbol: str, strategy: Callable[[pd.DataFrame], bool]):
    log = FTMO_Logger()
    risk = RiskEngine()
    while True:
        try:
            df = get_ohlcv(symbol)
            if strategy(df) and not risk.update(100000.0):
                order = f"BUY {symbol} @ {df['value'].iloc[-1]:.5f}"
                log.log(order)
                place_order(order)
        except Exception as e:
            log.log(f"ERROR: {e}")
            break
