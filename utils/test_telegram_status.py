#!/usr/bin/env python3
import os
import asyncio
import aiohttp
from dotenv import load_dotenv

load_dotenv("/opt/coreflow/.env")

BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

async def test_telegram():
    if not BOT_TOKEN or not CHAT_ID:
        print("❌ TELEGRAM_TOKEN oder CHAT_ID fehlt")
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": "✅ Telegram funktioniert (Test vom Watchdog)",
        "parse_mode": "HTML"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                status = resp.status
                body = await resp.text()
                print(f"📬 Status: {status}\n🔎 Antwort: {body}")
    except Exception as e:
        print(f"❌ Telegram-Verbindung fehlgeschlagen: {e}")

if __name__ == "__main__":
    asyncio.run(test_telegram())
