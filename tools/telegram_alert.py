import os
import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path="/opt/coreflow/.env")

TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "False") == "True"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
ALERT_LEVEL = os.getenv("TELEGRAM_ALERT_LEVEL", "normal")

def send_telegram_message(message, level="normal"):
    if not TELEGRAM_ENABLED:
        return
    if level == "low" and ALERT_LEVEL != "low":
        return
    if level == "normal" and ALERT_LEVEL == "high":
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        r = requests.post(url, data=payload)
        if r.status_code != 200:
            print(f"[!] Telegram-Fehler: {r.text}")
    except Exception as e:
        print(f"[!] Telegram-Sendeproblem: {e}")
