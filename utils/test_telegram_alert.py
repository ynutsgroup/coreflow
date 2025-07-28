#!/usr/bin/env python3
import sys
import asyncio
sys.path.insert(0, "/opt/coreflow/")
from utils.telegram_notifier import send_telegram_alert

async def main():
    await send_telegram_alert("✅ Telegram-Test erfolgreich!", alert_type="INFO")

if __name__ == "__main__":
    asyncio.run(main())
