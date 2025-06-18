#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoreFlow Signal Emitter (Linux) – REDIS Bridge für MT5 Windows
Vollständig verschlüsselt, FTMO-konform, hybridfähig

Pfad: /opt/coreflow/core/emitter/signal_emitter.py
"""

import os
import sys
import json
import redis
import logging
from datetime import datetime, timezone
from cryptography.fernet import Fernet

# ===== Logging-Konfiguration =====
LOG_PATH = '/opt/coreflow/logs/signal_emitter.log'
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SignalEmitter")

# ===== Verschlüsselte .env laden =====
try:
    from utils.decrypt_env import load_encrypted_env
except ImportError:
    logger.critical("Modul decrypt_env.py fehlt unter /opt/coreflow/utils/")
    sys.exit(1)

if not load_encrypted_env():
    logger.critical("❌ .env.enc konnte nicht geladen werden – Abbruch")
    sys.exit(1)

# ===== Fernet Schlüssel laden =====
try:
    with open('/opt/coreflow/infra/vault/encryption.key', 'rb') as f:
        key = f.read()
    fernet = Fernet(key)
except Exception as e:
    logger.critical(f"Fernet-Key-Fehler: {e}")
    sys.exit(1)

# ===== Redis-Verbindung aufbauen =====
try:
    r = redis.Redis(
        host=os.getenv("REDIS_HOST"),
        port=int(os.getenv("REDIS_PORT")),
        password=os.getenv("REDIS_PASSWORD"),
        ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
        decode_responses=False
    )
    r.ping()
    logger.info("✅ Redis-Verbindung erfolgreich")
except Exception as e:
    logger.critical(f"Redis-Verbindung fehlgeschlagen: {e}")
    sys.exit(1)

# ===== CLI Parameter validieren =====
if len(sys.argv) < 4:
    print("❌ Nutzung: python3 signal_emitter.py SYMBOL ACTION LOT [CONFIDENCE]")
    print("Beispiel: python3 signal_emitter.py BTCUSD BUY 0.05 0.91")
    sys.exit(1)

symbol = sys.argv[1].upper()
action = sys.argv[2].upper()
lot = float(sys.argv[3])
confidence = float(sys.argv[4]) if len(sys.argv) > 4 else float(os.getenv("MIN_CONFIDENCE", "0.9"))

# ===== Signal-Objekt erstellen =====
signal = {
    "symbol": symbol,
    "action": action,
    "lot": lot,
    "confidence": confidence,
    "timestamp": datetime.now(timezone.utc).isoformat()
}

# ===== Signal verschlüsseln und senden =====
try:
    encrypted = fernet.encrypt(json.dumps(signal).encode('utf-8'))
    channel = os.getenv("REDIS_CHANNEL", "trading_signals")
    r.publish(channel, encrypted)

    logger.info(f"📤 Signal gesendet: {symbol} {action} {lot} → Kanal '{channel}'")

except Exception as e:
    logger.critical(f"❌ Signalveröffentlichung fehlgeschlagen: {e}")
    sys.exit(1)
