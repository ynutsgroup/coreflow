#!/usr/bin/env python3
# ✅ Testskript für Telegram-Integration mit funktionierender .env

import os
import asyncio
from dotenv import load_dotenv
import aiohttp

# .env laden
load_dotenv(dotenv_path="/opt/coreflow/.env")

TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "False").lower() == "true"
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

async def send_telegram_message(message: str):
    if not TELEGRAM_ENABLED:
        print("⚠️ Telegram ist deaktiviert (TELEGRAM_ENABLED=False)")
        return

    if not BOT_TOKEN or not CHAT_ID:
        print("❌ TELEGRAM_TOKEN oder TELEGRAM_CHAT_ID fehlen.")
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                status = resp.status
                response = await resp.text()
                print(f"✅ Status: {status}")
                print(f"📬 Antwort: {response}")
    except Exception as e:
        print(f"❌ Fehler beim Senden: {e}")

if __name__ == "__main__":
    asyncio.run(send_telegram_message("✅ Telegram-Test: CoreFlow kann Nachrichten senden!"))
