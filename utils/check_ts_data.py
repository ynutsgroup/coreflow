import redis

r = redis.Redis(host="127.0.0.1", port=6380, password="ncasInitHW!")
result = r.execute_command("TS.RANGE", "EURUSD_ticks", "-", "+", "COUNT", 5)
print("🔍 Sample:", result)
