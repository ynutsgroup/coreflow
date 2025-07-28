import redis

# Direkte Redis-Verbindungsdaten
REDIS_HOST = 'os.getenv('REDIS_HOST')'
REDIS_PORT = int(os.getenv('REDIS_PORT'))
REDIS_PASSWORD = 'NeuesSicheresPasswort#2024!'

def test_redis_connection():
    try:
        # Verbindung zu Redis herstellen
        print(f"Verbindungsversuch zu Redis auf {REDIS_HOST}:{REDIS_PORT}...")
        client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True)

        # Testen, ob Redis läuft
        response = client.ping()
        print(f"Antwort von Redis: {response}")

        if response == "PONG":
            print("✅ Verbindung zu Redis erfolgreich!")

            # Test-Signal senden
            signal = {"symbol": "EURUSD", "action": "BUY", "volume": 0.2}
            client.publish('trading_signals', str(signal))
            print("✅ Test-Signal an 'trading_signals' gesendet")
        else:
            print("❌ Redis-Verbindung nicht erfolgreich")

    except redis.exceptions.ConnectionError as e:
        print(f"❌ Fehler bei der Redis-Verbindung: {e}")
    except Exception as e:
        print(f"❌ Allgemeiner Fehler: {e}")

if __name__ == "__main__":
    test_redis_connection()
