#!/usr/bin/env python3
"""
Telegram Notification Bot v2.0

Enhanced features:
- Robust error handling
- Detailed logging
- Configurable message formatting
- Async context manager for bot session
"""

import os
import asyncio
import logging
from typing import Optional
from dotenv import load_dotenv
from aiogram import Bot
from aiogram.exceptions import TelegramAPIError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TelegramNotifier')

class TelegramNotifier:
    def __init__(self):
        """Initialize the Telegram notifier"""
        load_dotenv()
        self.token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.bot = None
        
        if not self.token or not self.chat_id:
            raise ValueError("Missing Telegram configuration in .env file")

    async def __aenter__(self):
        """Async context manager entry"""
        self.bot = Bot(token=self.token)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.bot:
            await self.bot.close()

    async def send_message(
        self,
        message: str,
        parse_mode: Optional[str] = "Markdown",
        disable_notification: bool = False
    ) -> bool:
        """
        Send a message to Telegram with error handling
        
        Args:
            message: The message to send
            parse_mode: Markdown/HTML formatting
            disable_notification: Silent message
            
        Returns:
            bool: True if message was sent successfully
        """
        if not self.bot:
            raise RuntimeError("Bot not initialized. Use async context manager")

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode,
                disable_notification=disable_notification
            )
            logger.info("Message sent successfully")
            return True
            
        except TelegramAPIError as e:
            logger.error(f"Telegram API error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return False

async def main():
    """Main function to demonstrate usage"""
    try:
        async with TelegramNotifier() as notifier:
            # Example messages
            success = await notifier.send_message(
                "🚀 *System Notification*\n"
                "FTMO Risk Manager is running\n"
                "`Status: Operational`"
            )
            
            if not success:
                logger.warning("Failed to send notification")
                
    except Exception as e:
        logger.critical(f"Application error: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        raise
