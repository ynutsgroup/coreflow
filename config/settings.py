import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MT5_LOGIN = int(os.getenv("MT5_LOGIN", 0))
    MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
    MT5_SERVER = os.getenv("MT5_SERVER", "")
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
    SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 2))
    HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", 3600))
    MAX_SPREAD_RATIO = float(os.getenv("MAX_SPREAD_RATIO", 3.0))
    PRICE_CHANGE_THRESHOLD = float(os.getenv("PRICE_CHANGE_THRESHOLD", 0.0001))

config = Config()
