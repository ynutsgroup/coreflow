#!/usr/bin/env python3
# Async Telegram Notifier for aiogram/async apps

import os
from dotenv import load_dotenv
import aiohttp

load_dotenv("/opt/coreflow/.env")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

async def send_telegram_alert(message: str, alert_type: str = "INFO") -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
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
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            params = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
            async with session.get(url, params=params) as response:
                return response.status == 200
    except Exception as e:
        print(f"Async Telegram Error: {str(e)}")
        return False
