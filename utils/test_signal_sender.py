#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# CoreFlow Test Signal Sender v1.0

import os
import sys
import json
import redis
from datetime import datetime
from dotenv import load_dotenv
from cryptography.fernet import Fernet

# --- Load Configuration ---
BASE_DIR = "/opt/coreflow"
ENV_FILE = os.path.join(BASE_DIR, ".env")
if not os.path.exists(ENV_FILE):
    print(f"❌ .env file not found at {ENV_FILE}")
    sys.exit(1)

load_dotenv(ENV_FILE)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_CHANNEL = os.getenv("REDIS_CHANNEL", "trading_signals")
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")

if not ENCRYPTION_KEY:
    print("❌ ENCRYPTION_KEY missing in .env!")
    sys.exit(1)

cipher = Fernet(ENCRYPTION_KEY.encode())

try:
    redis_client = redis.StrictRedis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=True,
        socket_timeout=10,
        socket_keepalive=True
    )
    redis_client.ping()
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
    sys.exit(1)

# --- Build and Send Test Signal ---
def send_test_signal():
    signal = {
        "symbol": "BTCUSD",
        "action": "BUY",
        "price": 103500.0,
        "stop_loss": 50,
        "take_profit": 100,
        "confidence": 0.85,
        "strategy": "TEST_SIGNAL_V1",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "5.1"
    }

    try:
        payload = json.dumps(signal, ensure_ascii=False)
        encrypted_payload = cipher.encrypt(payload.encode()).decode()

        # Publish to Redis
        redis_client.publish(REDIS_CHANNEL, encrypted_payload)

        # Output Results
        print(f"✅ Test Signal Sent on Channel: {REDIS_CHANNEL}")
        print(f"🔐 Encrypted Payload:\n{encrypted_payload}\n")
        print(f"📖 Decrypted JSON:\n{json.dumps(signal, indent=4)}")

        return True

    except Exception as e:
        print(f"❌ Failed to send test signal: {str(e)}")
        return False

if __name__ == "__main__":
    send_test_signal()
