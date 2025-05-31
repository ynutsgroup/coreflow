#!/usr/bin/env python3
import os
import redis
import json
from dotenv import load_dotenv
from cryptography.fernet import Fernet, InvalidToken
import logging
from pathlib import Path
from typing import Optional

# .env laden
load_dotenv()

# Redis-Konfiguration
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_CHANNEL = os.getenv("REDIS_CHANNEL", "trading_signals")
REDIS_STREAM = os.getenv("REDIS_STREAM", "trading_signals")

# Verschlüsselungsschlüssel
FERNET_KEY_PATH = "/opt/coreflow/.env.key"
try:
    with open(FERNET_KEY_PATH, "rb") as f:
        fernet = Fernet(f.read())
except (FileNotFoundError, PermissionError) as e:
    logging.critical(f"Fernet-Key nicht gefunden oder Zugriffsfehler: {e}")
    exit(1)
except InvalidToken as e:
    logging.critical(f"Ungültiger Fernet-Key: {e}")
    exit(1)

# Logging einrichten
LOG_DIR = Path(os.getenv("LOG_DIR", "/opt/coreflow/logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "signal_aggregator.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("SignalAggregator")

def send_signal_to_redis(symbol: str, action: str, volume: float = 0.1) -> None:
    """Sendet ein Signal an Redis Stream"""
    signal = {
        "symbol": symbol.upper(),
        "action": action.upper(),
        "volume": round(volume, 2)
    }

    try:
        payload = json.dumps(signal).encode("utf-8")
        encrypted = fernet.encrypt(payload)
        
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            ssl=os.getenv("REDIS_SSL", "false").lower() == "true"
        )
        
        r.xadd(REDIS_STREAM, {"symbol": signal["symbol"], "encrypted": encrypted})
        logger.info(f"Signal gesendet: {signal['symbol']} {signal['action']} ({signal['volume']} Lot)")
    except (redis.RedisError, json.JSONDecodeError) as e:
        logger.error(f"Fehler beim Senden des Signals: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unerwarteter Fehler: {str(e)}", exc_info=True)
        raise

def main() -> None:
    """Main Loop"""
    try:
        send_signal_to_redis("EURUSD", "BUY", 0.1)
        send_signal_to_redis("GBPUSD", "SELL", 0.2)
    except Exception as e:
        logger.error(f"Fehler im Hauptloop: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
