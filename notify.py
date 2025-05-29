import os
import asyncio
from aiogram import Bot
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

async def send_telegram_message(text: str):
    bot = Bot(token=TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text=text)
    await bot.session.close()

def notify(text: str):
    asyncio.run(send_telegram_message(text))
