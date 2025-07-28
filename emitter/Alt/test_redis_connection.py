import os
import redis
from dotenv import load_dotenv

load_dotenv("/opt/coreflow/.env")

r = redis.Redis(
    host="127.0.0.1",
    port=6379,
    password=os.getenv("REDIS_PASSWORD")
)

try:
    pong = r.ping()
    print("✅ Redis connected:", pong)
    r.xadd("trading_signals", {"symbol": "EURUSD", "encrypted": b"TEST_SIGNAL"})
    print("📤 Test-Signal erfolgreich gesendet.")
except Exception as e:
    print("❌ Redis error:", e)
