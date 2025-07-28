#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CoreFlow TimeSeries Writer – Secure FTMO Edition
🔐 Lädt verschlüsselte Umgebungsvariablen via decrypt_env.py
📊 Schreibt Werte in RedisTimeSeries (TS.ADD)
"""

import os
import sys
import redis
import logging
from pathlib import Path

# 🔐 Dynamischer Pfad zum verschlüsselten .env-Loader
decrypt_path = Path("/opt/coreflow/utils")
if decrypt_path.exists():
    sys.path.insert(0, str(decrypt_path))
else:
    raise FileNotFoundError(f"❌ decrypt_env.py nicht gefunden: {decrypt_path}")

from decrypt_env import load_env, find_latest_env_enc

# 🔐 Neueste gültige .env.enc laden
env_path = find_latest_env_enc()
if not env_path:
    raise FileNotFoundError("❌ Keine gültige .env.enc-Datei gefunden.")
env = load_env(env_path, "/opt/coreflow/infra/vault/encryption.key")

# 🌐 Redis-Konfiguration aus Umgebungsvariablen
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
TS_KEY_TEMPLATE = os.getenv("REDIS_TS_TEMPLATE", "test:{symbol}")

# 📋 Logger-Konfiguration
logger = logging.getLogger("TimeSeriesWriter")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def write_signal_to_ts(symbol: str, value: float) -> bool:
    """Speichert einen Wert für ein gegebenes Symbol in RedisTimeSeries."""
    try:
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            decode_responses=True
        )

        key = TS_KEY_TEMPLATE.format(symbol=symbol.lower())

        # Erstelle TimeSeries-Key, falls er nicht existiert
        if not r.exists(key):
            logger.info(f"📊 Neuer TimeSeries-Key wird erstellt: {key}")
            r.execute_command("TS.CREATE", key)

        # Wert schreiben mit aktuellem Timestamp
        r.execute_command("TS.ADD", key, "*", value)
        logger.info(f"✅ {symbol.upper()} gespeichert: {value}")
        return True

    except redis.RedisError as e:
        logger.error(f"❌ Redis-Fehler beim Schreiben von {symbol}: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Allgemeiner Fehler: {e}")
        return False
