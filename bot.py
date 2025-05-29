import asyncio
import os
from aiogram import Bot, Dispatcher, Router
from aiogram.filters import Command
from aiogram.types import Message
from dotenv import load_dotenv
import psutil
from datetime import datetime

load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
bot = Bot(token=TOKEN)
dp = Dispatcher()
router = Router()
dp.include_router(router)

@router.message(Command("start"))
async def start_handler(message: Message):
    await message.answer("Hallo! Ich bin dein CoreFlow-Bot.")

@router.message(Command("status"))
async def status_handler(message: Message):
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    await message.answer(f"🕒 {now}\n💻 CPU: {cpu}%\n🧠 RAM: {mem}%")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
