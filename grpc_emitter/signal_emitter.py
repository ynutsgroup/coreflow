import redis
import json
import os
from dotenv import load_dotenv

# .env-Datei laden (z. B. Redis-Port und Passwort)
load_dotenv('/opt/coreflow/config/.env.grpc')

# Redis-Verbindung aufbauen
redis_client = redis.StrictRedis(
    host=os.getenv("REDIS_HOST", "127.0.0.1"),
    port=int(os.getenv("REDIS_PORT", 6380)),
    password=os.getenv("REDIS_PASSWORD", "ncasInitHW!"),
    decode_responses=True
)

def emit_signal(symbol: str, action: str):
    """
    Sendet ein Handelssignal als JSON an Redis-Queue 'coreflow:signals'
    """
    payload = {
        "symbol": symbol,
        "action": action
    }

    try:
        redis_client.lpush("coreflow:signals", json.dumps(payload))
        print(f"[ZMQ] Signal → Redis gepusht: {payload}")
    except Exception as e:
        print(f"[ZMQ ERROR] Redis Push fehlgeschlagen: {e}")
