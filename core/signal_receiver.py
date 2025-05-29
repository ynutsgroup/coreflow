#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# CoreFlow Signal Receiver v5.0 (FTMO/KI Ready)

import os
import sys
import json
import time
import redis
import MetaTrader5 as mt5
from dotenv import load_dotenv
from datetime import datetime
from cryptography.fernet import Fernet
import logging
import requests

# === Load Configuration ===
BASE_DIR = "/opt/coreflow"
ENV_FILE = os.path.join(BASE_DIR, ".env")

if not os.path.exists(ENV_FILE):
    print(f"❌ .env file not found at {ENV_FILE}")
    sys.exit(1)

load_dotenv(ENV_FILE)

# MT5 Config
TRADE_MODE = os.getenv("TRADE_MODE", "TEST").upper()
MT5_LOGIN = int(os.getenv(f"MT5_{TRADE_MODE}_LOGIN"))
MT5_PASSWORD = os.getenv(f"MT5_{TRADE_MODE}_PASSWORD")
MT5_SERVER = os.getenv(f"MT5_{TRADE_MODE}_SERVER")

# Redis Config
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_CHANNEL = os.getenv("REDIS_CHANNEL", "trading_signals")
REDIS_SSL = os.getenv("REDIS_SSL", "false").lower() == "true"

# Security
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
if not ENCRYPTION_KEY:
    print("❌ ENCRYPTION_KEY missing in .env!")
    sys.exit(1)
cipher = Fernet(ENCRYPTION_KEY.encode())

# Telegram Config
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "false").lower() == "true"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# === Logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# === Redis Connection ===
try:
    redis_client = redis.StrictRedis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        ssl=REDIS_SSL,
        decode_responses=True,
        socket_timeout=10,
        socket_keepalive=True
    )
    redis_client.ping()
except redis.ConnectionError as e:
    logging.critical(f"❌ Redis connection error: {e}")
    sys.exit(1)

# === MT5 Connection ===
if not mt5.initialize():
    logging.critical(f"❌ MT5 initialization failed: {mt5.last_error()}")
    sys.exit(1)

if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
    logging.critical(f"❌ MT5 login failed: {mt5.last_error()}")
    mt5.shutdown()
    sys.exit(1)

logging.info(f"✅ Connected to MT5 in {TRADE_MODE} mode")

# === Telegram Notification ===
def send_telegram(message):
    if not TELEGRAM_ENABLED:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        logging.warning(f"⚠️ Telegram error: {e}")

# === Signal Processor ===
def process_signal(data):
    try:
        decrypted = cipher.decrypt(data.encode()).decode()
        signal = json.loads(decrypted)

        required = ["symbol", "action", "price", "stop_loss", "take_profit", "confidence", "timestamp"]
        for field in required:
            if field not in signal:
                raise ValueError(f"Missing field: {field}")

        symbol = signal["symbol"]
        action = signal["action"].upper()
        price = float(signal["price"])
        stop_loss = int(signal["stop_loss"])
        take_profit = int(signal["take_profit"])
        confidence = float(signal["confidence"])

        if confidence < float(os.getenv("MIN_CONFIDENCE", 0.75)):
            logging.warning(f"⚠️ Confidence {confidence} below minimum threshold")
            return

        # Prepare MT5 Order
        lot_size = 0.1  # Default lot size; implement risk-based calculation if needed
        order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": price - (stop_loss * 0.0001) if action == "BUY" else price + (stop_loss * 0.0001),
            "tp": price + (take_profit * 0.0001) if action == "BUY" else price - (take_profit * 0.0001),
            "deviation": 10,
            "magic": 20240615,
            "comment": f"CFv5 | Confidence: {confidence:.2%}",
            "type_filling": mt5.ORDER_FILLING_FOK
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"Order failed: {result.comment}")

        msg = (
            f"✅ Trade Executed\n"
            f"Symbol: {symbol}\n"
            f"Action: {action}\n"
            f"Price: {price}\n"
            f"SL: {request['sl']}\n"
            f"TP: {request['tp']}\n"
            f"Lot: {lot_size}\n"
            f"Confidence: {confidence:.2%}"
        )
        logging.info(msg)
        send_telegram(msg)

    except Exception as e:
        logging.error(f"❌ Signal Processing Error: {e}")
        send_telegram(f"❌ Signal Processing Error: {e}")

# === Main Receiver Loop ===
if __name__ == "__main__":
    pubsub = redis_client.pubsub()
    pubsub.subscribe(REDIS_CHANNEL)
    logging.info(f"📡 Listening on Redis Channel: {REDIS_CHANNEL}")

    try:
        for message in pubsub.listen():
            if message["type"] == "message":
                process_signal(message["data"])
    except KeyboardInterrupt:
        logging.info("🛑 Graceful shutdown requested")
    finally:
        if mt5.initialized():
            mt5.shutdown()
        logging.info("🚪 Signal Receiver stopped")
