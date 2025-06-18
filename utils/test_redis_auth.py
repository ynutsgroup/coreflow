import redis

REDIS_HOST = '127.0.0.1'
REDIS_PORT = 6379
REDIS_PASSWORD = 'NeuesSicheresPasswort#2024!'

def test_redis_connection():
    try:
        client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True)
        response = client.ping()
        print(f"Antwort von Redis: {response}")
        if response:
            print("✅ Verbindung zu Redis erfolgreich!")
        else:
            print("❌ Redis-Verbindung nicht erfolgreich")
    except Exception as e:
        print(f"❌ Fehler bei der Redis-Verbindung: {e}")

if __name__ == "__main__":
    test_redis_connection()
