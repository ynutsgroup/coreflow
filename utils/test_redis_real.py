import redis
import time
from dotenv import load_dotenv
import os

# .env Datei laden, wenn du Umgebungsvariablen verwendest
load_dotenv()

# Redis-Verbindungsdaten aus der .env-Datei
REDIS_HOST = os.getenv('REDIS_HOST', '127.0.0.1')  # Standard: 127.0.0.1
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))  # Standard: 6379
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', 'NeuesSicheresPasswort#2024!')  # Standardpasswort

def test_redis():
    try:
        # Verbindung zu Redis herstellen
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            socket_timeout=5,
            socket_connect_timeout=5,
            decode_responses=True  # Ermöglicht die einfache Handhabung von Strings
        )

        # Test 1: Ping
        ping_ok = r.ping()
        print(f"PING: {'✅' if ping_ok else '❌'}")

        # Test 2: Daten schreiben/lesen
        test_key = f"coreflow_test_{int(time.time())}"
        r.set(test_key, "success", ex=10)  # Mit 10s TTL (Time To Live)
        value = r.get(test_key)
        print(f"DATA TEST: {'✅' if value == 'success' else '❌'}")

        # Test 3: Stream-Funktion
        try:
            r.xadd("test_stream", {"data": "test"})
            print("STREAM TEST: ✅")
        except Exception as e:
            print(f"STREAM TEST: ❌ ({str(e)})")

    except redis.exceptions.ConnectionError as e:
        print(f"❌ Redis-Verbindung konnte nicht hergestellt werden: {e}")
    except Exception as e:
        print(f"❌ Kritischer Fehler: {str(e)}")

# Aufruf der Funktion
test_redis()
