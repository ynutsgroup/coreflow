import redis
import time
import random

r = redis.Redis(host="127.0.0.1", port=6380, password="ncasInitHW!")

while True:
    price = round(random.uniform(1.0500, 1.0600), 5)
    timestamp = int(time.time() * 1000)
    r.execute_command("TS.ADD", "EURUSD_ticks", "*", price, "LABELS", "symbol", "eurusd")
    print(f"[+] Tick @ {timestamp}: {price}")
    time.sleep(5)
