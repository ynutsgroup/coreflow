#!/usr/bin/env python3
# /opt/coreflow/tests/btc_emitter_test.py
# Sendet BTCUSD BUY-Signal zur Validierung der Redis-Verbindung

import os
import json
import redis
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv
from cryptography.fernet import Fernet

# === .env laden ===
load_dotenv('/opt/coreflow/.env')

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('/opt/coreflow/logs/btc_emitter_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BTCEmitterTest")

# === Fernet Initialisierung ===
try:
    with open('/opt/coreflow/.env.key', 'rb') as f:
        fernet = Fernet(f.read())
except Exception as e:
    logger.critical(f"Fernet-Key-Fehler: {e}")
    exit(1)

# === Redis Initialisierung ===
try:
    r = redis.Redis(
        host=os.getenv("REDIS_HOST"),
        port=int(os.getenv("REDIS_PORT")),
        password=os.getenv("REDIS_PASSWORD"),
        ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
        decode_responses=False
    )
    r.ping()
except Exception as e:
    logger.critical(f"Redis-Verbindung fehlgeschlagen: {e}")
    exit(1)

# === Signal erzeugen ===
signal = {
    "symbol": "BTCUSD",
    "action": "BUY",
    "confidence": 0.93,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "volume": 0.05
}

try:
    encrypted = fernet.encrypt(json.dumps(signal).encode('utf-8'))
    channel = os.getenv("REDIS_CHANNEL")
    r.publish(channel, encrypted)
    logger.info(f"📤 BTCUSD BUY Testsignal veröffentlicht auf Redis-Kanal '{channel}'")
except Exception as e:
    logger.critical(f"Signalveröffentlichung fehlgeschlagen: {e}")
    exit(1)
