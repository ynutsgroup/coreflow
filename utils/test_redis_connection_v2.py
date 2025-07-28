import redis
import time

def test_connection():
    try:
        # Verbindung mit Timeout
        r = redis.Redis(
            host='os.getenv('REDIS_HOST')',
            port=int(os.getenv('REDIS_PORT')),
            socket_timeout=3,
            socket_connect_timeout=3
        )
        
        # Erweiterter Test
        test_key = f"coreflow_test_{int(time.time())}"
        r.set(test_key, "success", ex=10)
        value = r.get(test_key)
        
        if value == b"success":
            print("✅ Redis funktioniert einwandfrei")
            return True
        else:
            print("⚠️ Redis antwortet, aber Datenzugriff fehlgeschlagen")
            return False
            
    except redis.ConnectionError as e:
        print(f"❌ Verbindungsfehler: {str(e)}")
        print("Bitte prüfen Sie:")
        print("1. Ist Redis installiert? (redis-server --version)")
        print("2. Läuft der Service? (sudo systemctl status redis-server)")
        print("3. Hört Redis auf Port int(os.getenv('REDIS_PORT'))? (sudo ss -tulnp | grep int(os.getenv('REDIS_PORT')))")
        return False
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {str(e)}")
        return False

if __name__ == "__main__":
    test_connection()
