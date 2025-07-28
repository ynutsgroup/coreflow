#!/usr/bin/env python3
# CoreFlow ENV Emitter – sendet zentrale .env von Linux an Windows via Redis

import os
import redis
import hashlib
from datetime import datetime, timezone
from dotenv import load_dotenv

# === Konfiguration ===
ENV_PATH = "/opt/coreflow/.env"
load_dotenv(ENV_PATH)

REDIS_HOST = os.getenv("REDIS_HOST", "os.getenv('REDIS_HOST')")
REDIS_PORT = int(os.getenv("REDIS_PORT", "int(os.getenv('REDIS_PORT'))"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "NeuesSicheresPasswort#2024!")
REDIS_CHANNEL = "coreflow.env.sync"

def read_env_content():
    """Lädt .env-Datei und berechnet SHA256"""
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    sha256 = hashlib.sha256(content.encode()).hexdigest()
    return content, sha256

def send_env():
    """Sendet ENV-Inhalt via Redis-PUB"""
    content, hash_value = read_env_content()
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hash": hash_value,
        "content": content
    }

    r = redis.StrictRedis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=True
    )

    r.publish(REDIS_CHANNEL, str(payload))
    print(f"✅ .env gesendet an Redis-Kanal '{REDIS_CHANNEL}' mit SHA256: {hash_value}")

if __name__ == "__main__":
    send_env()
