#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoreFlow Linux Proxy - Redis & ZMQ Emitter
Sendet empfangene Signale an MT5 Windows per ZMQ PUSH
"""

import os
import json
import zmq
import redis
import logging
import requests
from cryptography.fernet import Fernet

# === Logging ===
LOG_PATH = "/opt/coreflow/logs/coreflow_main.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CoreFlow.Linux")

# === Telegram Notifier ===
class TelegramNotifier:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = all([self.token, self.chat_id])

    def send(self, message: str, level: str = "info"):
        if not self.enabled:
            return
        icons = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "success": "✅",
            "critical": "🆘"
        }
        try:
            requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": f"{icons.get(level, '')} {message}",
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True
                },
                timeout=5
            )
        except Exception as e:
    import traceback; traceback.print_exc()
            logger.error(f"Telegram send error: {str(e)}")

telegram = TelegramNotifier()

# === ENV-Pfade ===
KEY_PATH = "/opt/coreflow/infra/vault/encryption.key"
ENC_ENV_PATH = "/opt/coreflow/tmp/.env.enc"

# === ENV-Entschlüsselung ===
def load_encrypted_env():
    try:
        with open(KEY_PATH, 'rb') as f:
            key = f.read()
        fernet = Fernet(key)
        with open(ENC_ENV_PATH, 'rb') as f:
            encrypted = f.read()
        decrypted = fernet.decrypt(encrypted).decode()
        for line in decrypted.splitlines():
            if line.strip() and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()
        logger.info("✅ .env.enc geladen")
    except Exception as e:
    import traceback; traceback.print_exc()
        logger.critical(f"❌ ENV Fehler: {e}")
        exit(1)

load_encrypted_env()

# === Redis Verbindung ===
try:
    redis_conn = redis.Redis(
        host=os.getenv("REDIS_HOST"),
        port=int(os.getenv("REDIS_PORT", int(os.getenv('REDIS_PORT')))),
        password=os.getenv("REDIS_PASSWORD"),
        db=int(os.getenv("REDIS_DB", 0)),
        decode_responses=True
    )
    redis_conn.ping()
    logger.info("✅ Redis verbunden")
    telegram.send("Redis verbunden", "success")
except Exception as e:
    import traceback; traceback.print_exc()
    logger.critical(f"❌ Redis Verbindung fehlgeschlagen: {e}")
    telegram.send(f"Redis Verbindung fehlgeschlagen: {e}", "critical")
    exit(1)

# === ZMQ PUSH Socket (Windows MT5) ===
context = zmq.Context()
socket = context.socket(zmq.PUSH)
zmq_target = os.getenv("MT5_ZMQ_TARGET", "os.getenv('REDIS_HOST')")
zmq_port = int(os.getenv("MT5_ZMQ_PORT", 7777))
socket.connect(f"tcp://{zmq_target}:{zmq_port}")
logger.info(f"✅ Verbunden zu MT5 via ZMQ: {zmq_target}:{zmq_port}")

# === Redis Listener ===
pubsub = redis_conn.pubsub()
pubsub.psubscribe("*_signals")
logger.info("📡 Lausche Redis Signale ...")
telegram.send("Starte Redis Listener", "info")

try:
    for message in pubsub.listen():
        if message["type"] == "pmessage":
            try:
                signal = json.loads(message["data"])
                logger.info(f"📥 Signal empfangen: {signal}")
                socket.send_json(signal)
                logger.info("📤 Signal an MT5 weitergeleitet")
            except Exception as e:
    import traceback; traceback.print_exc()
                logger.error(f"⚠️ Fehler bei Signalverarbeitung: {e}")
                telegram.send(f"Signalfehler: {e}", "error")
except KeyboardInterrupt:
    logger.info("🛑 CoreFlow gestoppt")
    telegram.send("CoreFlow Linux Proxy gestoppt", "info")
