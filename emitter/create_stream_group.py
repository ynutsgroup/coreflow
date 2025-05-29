import redis
import os
from dotenv import load_dotenv

load_dotenv("/opt/coreflow/.env")

r = redis.Redis(
    host="127.0.0.1",
    port=6379,
    password=os.getenv("REDIS_PASSWORD")
)

stream = "trading_signals"
group = "coreflow_group"

try:
    r.xgroup_create(name=stream, groupname=group, id="0", mkstream=True)
    print(f"✅ Gruppe '{group}' für Stream '{stream}' erfolgreich erstellt.")
except redis.exceptions.ResponseError as e:
    if "BUSYGROUP" in str(e):
        print(f"ℹ️ Gruppe '{group}' existiert bereits.")
    else:
        raise
