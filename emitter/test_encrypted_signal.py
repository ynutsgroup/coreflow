import os
import redis
import json
from cryptography.fernet import Fernet
from dotenv import load_dotenv
from datetime import datetime

# .env laden
load_dotenv("/opt/coreflow/.env")

# Fernet-Key laden
fernet = Fernet(os.getenv("FERNET_KEY").encode())

# Redis verbinden
r = redis.Redis(
    host="127.0.0.1",
    port=6379,
    password=os.getenv("REDIS_PASSWORD")
)

# Signal vorbereiten
signal = {
    "symbol": "EURUSD",
    "action": "BUY",
    "volume": 0.2,
    "zeitstempel": datetime.utcnow().isoformat()
}

# JSON → verschlüsseln
payload = json.dumps(signal).encode("utf-8")
encrypted = fernet.encrypt(payload)

# An Redis senden
r.xadd("trading_signals", {
    "symbol": signal["symbol"],
    "encrypted": encrypted
})

print("✅ Verschlüsseltes Test-Signal gesendet.")
