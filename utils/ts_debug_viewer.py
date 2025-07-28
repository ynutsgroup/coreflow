#!/usr/bin/env python3
import redis

r = redis.Redis(host='127.0.0.1', port=6380, password='ncasInitHW!')
keys = r.execute_command("TS.QUERYINDEX", "*")

for key in keys:
    print(f"Key: {key.decode()}")
    info = r.execute_command("TS.INFO", key)
    print(f"  First Timestamp: {info[15]}")
    print(f"  Last Timestamp: {info[17]}")
    print(f"  Labels: {info[21]}")
