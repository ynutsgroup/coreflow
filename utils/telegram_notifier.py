#!/usr/bin/env python3
# CoreFlow Async Telegram Notifier – angepasst an bestehende .env

import os
from dotenv import load_dotenv
import aiohttp

# === .env laden ===
load_dotenv("/opt/coreflow/.env")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")             # <- exakt wie in deiner .env
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")         # <- exakt wie in deiner .env

async def send_telegram_alert(message: str, alert_type: str = "INFO") -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Telegram-Konfiguration fehlt – TOKEN oder CHAT_ID nicht gefunden.")
        return False

    icon_map = {
        "INFO": "ℹ️",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "TRADE": "💱",
        "RISK": "🚨"
    }
    icon = icon_map.get(alert_type.upper(), "🔔")
    text = f"<b>{icon} {alert_type.upper()}</b>\n\n{message}"

    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            params = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "HTML"
            }
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return True
                else:
                    print(f"❌ Telegram API-Fehler: HTTP {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Async Telegram Exception: {str(e)}")
        return False
