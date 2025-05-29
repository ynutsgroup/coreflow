import os
import redis
import requests
from dotenv import load_dotenv
from pathlib import Path

class DiagnosticCommander:
    def __init__(self):
        self._load_config()

    def _load_config(self):
        env_path = Path(__file__).parent.parent / '.env'
        load_dotenv(dotenv_path=env_path)

        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_password = os.getenv("REDIS_PASSWORD")
        self.telegram_token = os.getenv("TELEGRAM_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.system_mode = os.getenv("SYSTEM_MODE", "UNDEFINED")
        self.dummy_mode = os.getenv("DUMMY_MODE", "UNDEFINED")

    def check_redis(self) -> bool:
        try:
            r = redis.StrictRedis(
                host=self.redis_host,
                port=self.redis_port,
                password=self.redis_password,
                decode_responses=True,
                socket_timeout=5
            )
            return r.ping()
        except Exception as e:
            print(f"❌ Redis Error: {e}")
            return False

    def check_telegram(self, message="✅ CoreFlow Diagnostic Commander Test") -> bool:
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            response = requests.post(
                url, data={"chat_id": self.telegram_chat_id, "text": message}, timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Telegram Error: {e}")
            return False

    def display_modes(self):
        print(f"SYSTEM_MODE: {self.system_mode}")
        print(f"DUMMY_MODE: {self.dummy_mode}")

if __name__ == "__main__":
    print("🚀 Starting CoreFlow Diagnostic Commander...\n")
    diag = DiagnosticCommander()

    print("🛟 Redis Check:")
    print("✅ Redis connection successful" if diag.check_redis() else "❌ Redis connection failed")

    print("\n📱 Telegram Check:")
    print("✅ Telegram test sent successfully" if diag.check_telegram() else "❌ Telegram test failed")

    print("\n⚙️ Current Modes:")
    diag.display_modes()

    print("\n✅ System Check completed.")
