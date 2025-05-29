#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# CoreFlow Signal Emitter – Fernet-kompatibel mit Receiver

import os
import json
import time
import hmac
import redis
from cryptography.fernet import Fernet
from dotenv import load_dotenv

# === Lade Umgebungsvariablen (.env) ===
load_dotenv("/opt/coreflow/.env")

# === Fester Fernet-Key (MUSS zum Windows-Receiver passen) ===
fernet = Fernet(b"MCKws5DhCg4Kza4RwCksvwXJLJXwY_fYdoSo7QaXHko=")

# === Redis-Verbindung ===
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "127.0.0.1"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=False
)

# === Beispielsignal ===
data = {
    "symbol": "BTCUSD",
    "action": "BUY",
    "volume": 0.05
}

# === HMAC-Signatur ===
hmac_key = os.getenv("SIGNAL_HMAC_KEY", "supersecretkey").encode()
signature = hmac.new(hmac_key, json.dumps(data).encode(), 'sha256').hexdigest()

# ✅ === RICHTIGES SIGNALFORMAT FÜR DEN RECEIVER ===
message = {
    "timestamp": time.time(),
    "data": data,
    "signature": signature
}

# === Verschlüsseln ===
encrypted = fernet.encrypt(json.dumps(message).encode())

# === An Redis senden ===
channel = os.getenv("REDIS_CHANNEL", "trading_signals")
redis_client.publish(channel, encrypted)

print("✅ Signal gesendet an Redis-Channel:", channel)
	
