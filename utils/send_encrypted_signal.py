import redis
import json
from cryptography.fernet import Fernet

KEY_PATH = "/opt/coreflow/.env.key"

with open(KEY_PATH, "rb") as f:
    key = f.read()
fernet = Fernet(key)

signal = {
    "symbol": "BTCUSD",
    "action": "BUY",
    "volume": 0.1
}
payload = json.dumps(signal).encode("utf-8")
encrypted_payload = fernet.encrypt(payload)

r = redis.Redis(
    host="127.0.0.1",
    port=6379,
    password="NeuesSicheresPasswort#2024!"
)
r.publish("trading_signals", encrypted_payload)

print("✅ Encrypted signal sent to Redis.")

