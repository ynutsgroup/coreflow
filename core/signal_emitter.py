import redis
import json
import os
from dotenv import load_dotenv

# ENV-Datei laden
load_dotenv('/opt/coreflow/config/.env.bridge')

def emit_signal(symbol: str, action: str):
    redis_host = os.getenv("REDIS_HOST")
    redis_port = int(os.getenv("REDIS_PORT"))
    redis_password = os.getenv("REDIS_PASSWORD")

    if not redis_host or not redis_port or not redis_password:
        raise ValueError("❌ Redis-Verbindungsdaten fehlen in .env.bridge")

    r = redis.StrictRedis(
        host=redis_host,
        port=redis_port,
        password=redis_password,
        decode_responses=True
    )

    payload = {
        "symbol": symbol,
        "action": action
    }

    r.lpush("coreflow:signals", json.dumps(payload))
    print(f"[SIGNAL] Gesendet an Redis: {payload}")
