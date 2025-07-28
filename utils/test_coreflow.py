import os
import redis
import requests
import logging
from dotenv import load_dotenv
from typing import Optional, Dict

# ✅ Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

class SystemChecker:
    def __init__(self, env_path: Optional[str] = None):
        self._load_config(env_path)
        
    def _load_config(self, env_path: Optional[str]):
        """Load and validate configuration"""
        env_file = env_path or os.path.join(os.getcwd(), '.env')
        if not os.path.isfile(env_file):
            logging.warning(f"⚠️ .env Datei nicht gefunden unter: {env_file}")
        load_dotenv(dotenv_path=env_file)
        
        self.redis_host = os.getenv("REDIS_HOST")
        self.redis_port = int(os.getenv("REDIS_PORT", "int(os.getenv('REDIS_PORT'))"))
        self.redis_password = os.getenv("REDIS_PASSWORD")
        self.telegram_token = os.getenv("TELEGRAM_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.system_mode = os.getenv("SYSTEM_MODE", "UNKNOWN")
        self.dummy_mode = os.getenv("DUMMY_MODE", "UNKNOWN")
        
        if not all([self.redis_host, self.telegram_token, self.telegram_chat_id]):
            raise ValueError("❌ Fehlende erforderliche Umgebungsvariablen in der .env Datei!")

    def check_redis(self) -> bool:
        """Check Redis connection"""
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
            logging.error(f"❌ Redis Verbindung fehlgeschlagen: {e}")
            return False

    def check_telegram(self, test_msg: str = "✅ CoreFlow Telegram Test") -> bool:
        """Check Telegram notification"""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            response = requests.post(
                url,
                data={"chat_id": self.telegram_chat_id, "text": test_msg},
                timeout=10
            )
            if response.status_code == 200:
                return True
            else:
                logging.error(f"❌ Telegram Antwort: {response.status_code} | {response.text}")
                return False
        except Exception as e:
            logging.error(f"❌ Telegram Fehler: {e}")
            return False

    def check_modes(self) -> Dict[str, str]:
        """Get system modes"""
        return {
            "SYSTEM_MODE": self.system_mode,
            "DUMMY_MODE": self.dummy_mode
        }

if __name__ == "__main__":
    logging.info("🔍 Starte CoreFlow System-Check...\n")
    try:
        checker = SystemChecker()
    except ValueError as ve:
        logging.critical(str(ve))
        exit(1)
    
    # ✅ Redis Check
    if checker.check_redis():
        logging.info("✅ Redis Verbindung erfolgreich.")
    else:
        logging.warning("❌ Redis Verbindung fehlgeschlagen.")

    # ✅ Telegram Check
    if checker.check_telegram():
        logging.info("✅ Telegram Test erfolgreich gesendet.")
    else:
        logging.warning("❌ Telegram Test fehlgeschlagen.")

    # ✅ System Modes
    modes = checker.check_modes()
    logging.info("\n🛠️ Aktuelle Systemmodi:")
    for key, value in modes.items():
        logging.info(f"🔹 {key}: {value}")

    logging.info("\n✅ System-Check abgeschlossen.")
