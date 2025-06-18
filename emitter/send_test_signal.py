#!/usr/bin/env python3
# send_test_signal.py – sendet verschlüsseltes Signal an Windows-MT5 via Redis

import os
import json
import redis
from cryptography.fernet import Fernet
from dotenv import load_dotenv

# === ENV laden ===
env_path = "/opt/coreflow/.env"
load_dotenv(env_path)

# === Konfiguration ===
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_CHANNEL = os.getenv("REDIS_CHANNEL", "trading_signals")
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")

# === Signal ===
signal_data = {
    "symbol": "EURUSD",
    "action": "BUY",
    "risk_percent": 1.0
}

# === Verschlüsselung ===
cipher = Fernet(ENCRYPTION_KEY.encode())
encrypted = cipher.encrypt(json.dumps(signal_data).encode())

# === Redis senden ===
r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    socket_timeout=5
)

r.publish(REDIS_CHANNEL, encrypted)
print("✅ Signal sent")
