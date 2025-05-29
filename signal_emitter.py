#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# CoreFlow SignalEmitter → Redis Secure Hybrid

import os
import json
import redis
import logging
import time
from dotenv import load_dotenv
from cryptography.fernet import Fernet
import requests

# === .env laden ===
load_dotenv()

# === Logging ===
LOG_DIR = os.getenv("LOG_DIR", "/opt/coreflow/logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "signal_emitter.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("SignalEmitter")

# === Redis-Konfig ===
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_CHANNEL = os.getenv("REDIS_CHANNEL", "trading_signals")

# === Telegram-Konfig ===
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "False") == "True"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# === Fernet-Key ===
FERNET_KEY_PATH = "/opt/coreflow/.env.key"
try:
    with open(FERNET_KEY_PATH, "rb") as f:
        fernet = Fernet(f.read())
except Exception as e:
    logger.critical(f"Fernet-Key nicht gefunden oder ungültig: {e}")
    exit(1)

# === Telegram-Benachrichtigung ===
def notify_telegram(text):
    if TELEGRAM_ENABLED and TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": text},
                timeout=3
            )
        except Exception as e:
            logger.warning(f"Telegram-Fehler: {e}")

# === Signal senden ===
def send_signal(symbol: str, action: str, volume: float = 0.1):
    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        ssl=os.getenv("REDIS_SSL", "false").lower() == "true"
    )

    signal = {
        "symbol": symbol.upper(),
        "action": action.upper(),
        "volume": round(volume, 2)
    }

    try:
        payload = json.dumps(signal).encode("utf-8")
        encrypted = fernet.encrypt(payload)
        r.publish(REDIS_CHANNEL, encrypted)

        msg = f"📢 Signal gesendet: {signal['symbol']} {signal['action']} ({signal['volume']} Lot)"
        logger.info(msg)
        notify_telegram(msg)
        print(msg)

    except Exception as e:
        err = f"❌ Fehler beim Senden: {e}"
        logger.error(err)
        notify_telegram(err)
        print(err)

# === Beispiel-Nutzung ===
if __name__ == "__main__":
    # Beispiel: EURUSD BUY 0.2 Lots
    send_signal("EURUSD", "BUY", 0.2)
