#!/usr/bin/env python3
# CoreFlow SignalEmitter – Dynamisch | FTMO-Ready | Linux

import os
import sys
import json
import redis
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv
from cryptography.fernet import Fernet
from pathlib import Path

# === .env laden ===
ENV_PATH = "/opt/coreflow/.env.enc"
KEY_PATH = "/opt/coreflow/infra/vault/encryption.key"
load_dotenv()

# === Logging konfigurieren ===
LOG_PATH = '/opt/coreflow/logs/signal_emitter.log'
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SignalEmitter")

# === .env entschlüsseln ===
def decrypt_env(enc_path: str, key_path: str) -> dict:
    try:
        with open(key_path, 'rb') as f:
            key = f.read()
        fernet = Fernet(key)

        with open(enc_path, 'rb') as ef:
            encrypted = ef.read()
        decrypted = fernet.decrypt(encrypted).decode()
        lines = decrypted.strip().split('\n')
        return {k: v for k, v in [line.strip().split('=', 1) for line in lines if '=' in line]}
    except Exception as e:
        logger.critical(f"❌ Fehler beim Entschlüsseln der .env: {e}")
        sys.exit(1)

env_vars = decrypt_env(ENV_PATH, KEY_PATH)

# === Redis Verbindung ===
try:
    redis_client = redis.StrictRedis(
        host=env_vars.get("REDIS_HOST", "localhost"),
        port=int(env_vars.get("REDIS_PORT", 6379)),
        password=env_vars.get("REDIS_PASSWORD", None),
        decode_responses=True
    )
    redis_client.ping()
    logger.info("✅ Redis-Verbindung erfolgreich")
except Exception as e:
    logger.critical(f"❌ Redis-Verbindungsfehler: {e}")
    sys.exit(1)

# === Eingabe prüfen ===
if len(sys.argv) < 5:
    logger.error("❌ Nutzung: signal_emitter.py SYMBOL ACTION LOT CONFIDENCE")
    sys.exit(1)

symbol = sys.argv[1].upper()
action = sys.argv[2].upper()
lot = float(sys.argv[3])
confidence = float(sys.argv[4])

# === Redis-Channel dynamisch je Symbol ===
channel = f"{symbol.lower()}_signals"

# === Signal erstellen ===
signal = {
    "symbol": symbol,
    "action": action,
    "lot": lot,
    "confidence": confidence,
    "timestamp": datetime.now(timezone.utc).isoformat()
}

# === Senden ===
try:
    redis_client.publish(channel, json.dumps(signal))
    logger.info(f"📤 Signal gesendet → {channel}: {signal}")
except Exception as e:
    logger.error(f"❌ Fehler beim Senden des Signals: {e}")
    sys.exit(1)
