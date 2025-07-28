#!/usr/bin/env python3

import redis
import os
from pathlib import Path

# === HARDCODED FALLBACK ===
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 6380
AUTH_FILE = "/opt/coreflow/secrets/redis.pass"

# === DYNAMISCH MIT FALLBACK ===
REDIS_HOST = os.getenv("REDIS_HOST", DEFAULT_HOST)
REDIS_PORT = int(os.getenv("REDIS_PORT", DEFAULT_PORT))

# === Passwort aus Datei lesen ===
if not Path(AUTH_FILE).exists():
    print(f"❌ Passwortdatei fehlt: {AUTH_FILE}")
    exit(1)

REDIS_PASSWORD = Path(AUTH_FILE).read_text().strip()

# === Verbindung testen ===
def test_redis_connection():
    try:
        client = redis.StrictRedis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            decode_responses=True,
            socket_timeout=3
        )
        response = client.ping()
        if response:
            print(f"✅ Verbindung zu Redis erfolgreich (Host: {REDIS_HOST}, Port: {REDIS_PORT})")
        else:
            print("❌ Redis hat auf PING nicht reagiert.")
    except redis.AuthenticationError:
        print("❌ Redis AUTH fehlgeschlagen – Passwort falsch?")
    except Exception as e:
        print(f"❌ Fehler bei der Redis-Verbindung: {e}")

if __name__ == "__main__":
    test_redis_connection()
