#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis TimeSeries Initializer – CoreFlow Institutional
📌 Erstellt symbolbasierte RedisTimeSeries mit Labels für TS.MRANGE Abfragen
"""

import redis
import os
from cryptography.fernet import Fernet
from dotenv import load_dotenv
from decrypt_env import load_env, find_latest_env_enc

# === ENV laden ===
env_path = find_latest_env_enc()
if not env_path:
    raise FileNotFoundError("❌ Keine gültige .env.enc-Datei gefunden.")
env = load_env(env_path, "/opt/coreflow/infra/vault/encryption.key")

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6380))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True
)

def init_ts_stream(symbol: str):
    key = f"{symbol.upper()}_ticks"
    print(f"🔧 Initialisiere Stream: {key}")

    # Wenn existiert, löschen
    if r.exists(key):
        r.delete(key)

    # Anlegen mit Label
    response = r.execute_command(
        "TS.CREATE", key,
        "RETENTION", 0,
        "LABELS", "symbol", symbol.lower()
    )
    print(f"✅ Erstellt: {key} mit Label symbol={symbol.lower()}")

if __name__ == "__main__":
    for sym in ["EURUSD", "BTCUSD", "XAUUSD", "NAS100", "GER40"]:
        init_ts_stream(sym)
