import requests
import logging
from config.settings import config

def send_telegram_message(message: str, important: bool = True):
    """Sendet eine Nachricht an den konfigurierten Telegram-Chat."""
    url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": config.TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200 and important:
            logging.warning(f"⚠️ Telegram Error: {response.status_code} – {response.text}")
    except Exception as e:
        if important:
            logging.warning(f"❌ Telegram send failed: {str(e)}")
