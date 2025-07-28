import redis
import json
from redis.exceptions import AuthenticationError, RedisError

def send_signal(env, signal):
    try:
        if "REDIS_URL" in env:
            print("✅ Redis via URL:", env["REDIS_URL"])
            r = redis.Redis.from_url(env["REDIS_URL"])
        else:
            print("⚠️ Redis via HOST/PORT")
            r = redis.Redis(
                host=env["REDIS_HOST"],
                port=int(env["REDIS_PORT"]),
                password=env.get("REDIS_PASSWORD", None),
                ssl=env.get("REDIS_SSL", "False").lower() == "true",
                socket_timeout=float(env.get("REDIS_SOCKET_TIMEOUT", 5))
            )

        channel = env.get("REDIS_CHANNEL", "coreflow:signals")
        payload = json.dumps(signal)
        r.lpush(channel, payload)
        return channel

    except AuthenticationError:
        raise Exception("Redis-Verbindungsfehler: Authentication required.")
    except RedisError as e:
        raise Exception(f"Redis-Fehler: {str(e)}")
    except Exception as e:
        raise Exception(f"Allgemeiner Fehler beim Redis-Zugriff: {str(e)}")
