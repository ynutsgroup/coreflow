#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoreFlow Signal Emitter – Hybrid Secure Version (v4.0)
Pfad: /opt/coreflow/core/emitter/signal_emitter.py
- Multi-Symbol Redis-Kanal
- Verschlüsselter Payload (Fernet)
- .env.enc-Handling mit decrypt_env.py
- FTMO-Konformität
"""

import os
import sys
import json
import redis
import logging
from datetime import datetime, timezone
from cryptography.fernet import Fernet

# === Logging ===
LOG_PATH = '/opt/coreflow/logs/signal_emitter.log'
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-14s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SignalEmitter")

# === Fix: Import decrypt_env.py aus utils ===
sys.path.append('/opt/coreflow/utils')
try:
    from decrypt_env import load_encrypted_env
except ImportError:
    logger.critical("❌ Modul 'decrypt_env.py' fehlt unter /opt/coreflow/utils/")
    sys.exit(1)

# === .env entschlüsseln ===
if not load_encrypted_env():
    logger.critical("❌ .env.enc konnte nicht geladen werden – Abbruch")
    sys.exit(1)

# === Fernet-Schlüssel laden ===
try:
    with open(os.getenv("FERNET_KEY_PATH", "/opt/coreflow/infra/vault/encryption.key"), 'rb') as f:
        key = f.read()
    fernet = Fernet(key)
except Exception as e:
    logger.critical(f"❌ Fernet-Key konnte nicht geladen werden: {e}")
    sys.exit(1)

# === Redis-Verbindung ===
try:
    r = redis.Redis(
        host=os.getenv("REDIS_HOST"),
        port=int(os.getenv("REDIS_PORT")),
        password=os.getenv("REDIS_PASSWORD"),
        ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
        decode_responses=False  # für verschlüsselte Bytes
    )
    r.ping()
    logger.info("✅ Redis-Verbindung erfolgreich")
except Exception as e:
    logger.critical(f"❌ Redis-Verbindung fehlgeschlagen: {e}")
    sys.exit(1)

# === CLI-Parameter prüfen ===
if len(sys.argv) < 4:
    logger.error("❌ Nutzung: python3 signal_emitter.py SYMBOL ACTION LOT [CONFIDENCE]")
    sys.exit(1)

symbol = sys.argv[1].upper()
action = sys.argv[2].upper()
lot = float(sys.argv[3])
confidence = float(sys.argv[4]) if len(sys.argv) > 4 else float(os.getenv("MIN_CONFIDENCE", "0.9"))

# === Signal vorbereiten ===
signal = {
    "symbol": symbol,
    "action": action,
    "lot": lot,
    "confidence": confidence,
    "timestamp": datetime.now(timezone.utc).isoformat()
}

# === Signal verschlüsseln und senden ===
try:
    payload = json.dumps(signal).encode('utf-8')
    encrypted = fernet.encrypt(payload)

    # Dynamischer Kanal: EURUSD → eurusd_signals
    channel = f"{symbol.lower()}_signals"
    r.publish(channel, encrypted)

    logger.info(f"📤 Signal gesendet → {channel}: {signal}")

except Exception as e:
    logger.critical(f"❌ Fehler beim Senden: {e}")
    sys.exit(1)
