#!/usr/bin/env python3
# CoreFlow Hybrid Main – Institutional AI Edition v4.1

import os, json, logging, threading, redis, zmq, time, requests
import numpy as np
from datetime import datetime
from cryptography.fernet import Fernet

# === Optionale KI-Module ===
AI_ENABLED = os.getenv("AI_ENABLED", "false").lower() == "true"
if AI_ENABLED:
    import torch
    from transformers import pipeline
    from sklearn.ensemble import IsolationForest

# === Logging Setup ===
LOG_PATH = "/opt/coreflow/logs/coreflow_main_hybrid.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CoreFlow.Hybrid")

# === Telegram Notifier ===
class TelegramNotifier:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = all([self.token, self.chat_id])

    def send(self, msg: str, level="info"):
        if not self.enabled:
            return
        icons = {
            "info": "ℹ️", "success": "✅", "warning": "⚠️",
            "error": "❌", "critical": "🆘", "ai": "🤖", "alert": "🚨"
        }
        try:
            requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": f"{icons.get(level, '')} {msg}",
                    "parse_mode": "HTML"
                },
                timeout=5
            )
        except Exception as e:
            logger.error(f"Telegram error: {e}")

# === ENV Loader ===
def load_env():
    try:
        key_path = "/opt/coreflow/infra/vault/encryption.key"
        enc_path = "/opt/coreflow/tmp/.env.enc"
        with open(key_path, "rb") as f:
            key = f.read()
        with open(enc_path, "rb") as f:
            encrypted = f.read()
        decrypted = Fernet(key).decrypt(encrypted).decode()
        for line in decrypted.splitlines():
            if line.strip() and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()
        logger.info("✅ ENV geladen")
    except Exception as e:
        logger.critical(f"❌ ENV Fehler: {e}")
        exit(1)

# === KI-Modul ===
class AISignalProcessor:
    def __init__(self):
        self.nlp = pipeline("text-classification", model="ProsusAI/finbert",
                            device=0 if torch.cuda.is_available() else -1)
        self.anomaly = IsolationForest(n_estimators=100, contamination=0.01)
        self.history = np.empty((0, 4))  # OHLC
        self.daily_orders = 0
        self.max_orders = int(os.getenv("FTMO_MAX_ORDERS", 100))

    def process(self, signal):
        result = signal.copy()
        if "news" in signal:
            try:
                sentiment = self.nlp(signal["news"])[0]
                result["sentiment"] = sentiment
            except Exception as e:
                logger.warning(f"NLP Error: {e}")

        if "ohlc" in signal:
            try:
                ohlc = np.array([
                    signal["ohlc"]["open"], signal["ohlc"]["high"],
                    signal["ohlc"]["low"], signal["ohlc"]["close"]
                ]).reshape(1, -1)
                self.history = np.vstack([self.history, ohlc])[-500:]
                result["is_anomaly"] = int(self.anomaly.fit_predict(ohlc)[0])
            except Exception as e:
                logger.warning(f"Anomaly Error: {e}")

        if self.daily_orders >= self.max_orders:
            result["ftmo_compliant"] = False
        else:
            result["ftmo_compliant"] = True
            self.daily_orders += 1

        return result

# === Setup ===
load_env()
telegram = TelegramNotifier()
telegram.send("🤖 CoreFlow Hybrid gestartet", "ai")
ai = AISignalProcessor() if AI_ENABLED else None

# === Redis + ZMQ ===
try:
    redis_conn = redis.Redis.from_url(os.getenv("REDIS_URL"))
    redis_conn.ping()
    logger.info("✅ Redis verbunden")
except Exception as e:
    logger.critical(f"❌ Redis Fehler: {e}")
    telegram.send(f"Redis Fehler: {e}", "critical")
    exit(1)

try:
    zmq_context = zmq.Context()
    zmq_socket = zmq_context.socket(zmq.PUSH)
    zmq_target = os.getenv("MT5_ZMQ_TARGET", "127.0.0.1")
    zmq_port = int(os.getenv("MT5_ZMQ_PORT", 7777))
    zmq_socket.connect(f"tcp://{zmq_target}:{zmq_port}")
    logger.info(f"✅ ZMQ verbunden mit {zmq_target}:{zmq_port}")
except Exception as e:
    logger.critical(f"❌ ZMQ Fehler: {e}")
    telegram.send(f"ZMQ Fehler: {e}", "critical")
    exit(1)

# === Redis Listener ===
def redis_listener():
    pubsub = redis_conn.pubsub()
    pubsub.psubscribe("*_signals")
    logger.info("📡 Lausche Redis-Signale ...")
    for message in pubsub.listen():
        if message["type"] == "pmessage":
            try:
                signal = json.loads(message["data"])
                if AI_ENABLED and ai:
                    signal = ai.process(signal)
                zmq_socket.send_json(signal)
                logger.info(f"📤 Signal weitergeleitet: {signal}")
            except Exception as e:
                logger.error(f"⚠️ Signal Fehler: {e}")
                telegram.send(f"Signal Fehler: {e}", "error")

# === Starten ===
threading.Thread(target=redis_listener, daemon=True).start()

# === Health Check Loop ===
while True:
    try:
        if not redis_conn.ping():
            raise Exception("Redis nicht erreichbar")
        telegram.send("✅ Health Check OK", "info")
    except Exception as e:
        logger.error(f"Health Check Fehler: {e}")
        telegram.send("Health Check Fehler", "critical")
        exit(1)
    time.sleep(300)
