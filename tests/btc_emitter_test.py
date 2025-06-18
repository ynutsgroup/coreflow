#!/usr/bin/env python3
# /opt/coreflow/tests/btc_emitter_test.py
# Sendet ein verschlüsseltes BTCUSD BUY-Testsignal über Redis (kompatibel mit CoreFlow MT5 Receiver)

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
LOG_PATH = '/opt/coreflow/logs/btc_emitter_test.log'
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BTCEmitterTest")

# === Fernet Initialisierung ===
try:
    with open('/opt/coreflow/.env.key', 'rb') as f:
        key = f.read()
    fernet = Fernet(key)
except Exception as e:
    logger.critical(f"❌ Fernet-Key-Fehler: {e}")
    exit(1)

# === Redis Initialisierung ===
try:
    r = redis.Redis(
        host=os.getenv("REDIS_HOST"),
        port=int(os.getenv("REDIS_PORT")),
        password=os.getenv("REDIS_PASSWORD"),
        ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
        decode_responses=False  # Senden von verschlüsselten BYTES
    )
    r.ping()
except Exception as e:
    logger.critical(f"❌ Redis-Verbindung fehlgeschlagen: {e}")
    exit(1)

# === Testsignal erzeugen ===
signal = {
    "symbol": "BTCUSD",
    "action": "BUY",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "lot": 0.05,
    "confidence": 0.93
}

try:
    encrypted = fernet.encrypt(json.dumps(signal).encode('utf-8')).decode('utf-8')
    wrapped = json.dumps({ "encrypted_payload": encrypted })
    channel = os.getenv("REDIS_CHANNEL", "trading_signals")

    r.publish(channel, wrapped.encode('utf-8'))
    logger.info(f"📤 BTCUSD BUY Testsignal veröffentlicht auf Redis-Kanal '{channel}'")

except Exception as e:
    logger.critical(f"💥 Signalveröffentlichung fehlgeschlagen: {e}")
    exit(1)
