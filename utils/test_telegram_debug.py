import os
import asyncio
import aiohttp
from dotenv import load_dotenv

load_dotenv("/opt/coreflow/.env")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

async def send_debug_message():
    if not BOT_TOKEN or not CHAT_ID:
        print("❌ BOT_TOKEN oder CHAT_ID fehlen.")
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": "✅ Testnachricht von CoreFlow (Debug-Modus)",
        "parse_mode": "HTML"
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=payload) as response:
            status = response.status
            content = await response.text()
            print(f"➡️ Status: {status}")
            print(f"➡️ Antwort: {content}")

asyncio.run(send_debug_message())
