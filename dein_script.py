import os
from dotenv import load_dotenv
from aiogram import Bot
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from aiogram.types import ParseMode  # Prüfen, ob das korrekt funktioniert
import asyncio

# Lade Umgebungsvariablen aus der .env-Datei
load_dotenv()

# Holen der Telegram-Daten aus der .env-Datei
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Telegram-Bot initialisieren
bot = Bot(token=TELEGRAM_TOKEN)

# Funktion zum Senden von Telegram-Nachrichten
async def send_telegram_message(message: str):
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode=ParseMode.MARKDOWN)
        print("Nachricht erfolgreich gesendet!")
    except Exception as e:
        print(f"Fehler beim Senden der Nachricht: {e}")

# Testfunktion für das Senden von Nachrichten
async def main():
    # Beispielnachricht
    message = "Dies ist eine Testnachricht von deinem Trading-Bot!"
    
    # Nachricht senden
    await send_telegram_message(message)

# Ausführen der Nachrichtensendung
if __name__ == "__main__":
    asyncio.run(main())
