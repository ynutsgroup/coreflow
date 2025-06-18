#!/usr/bin/env python3
# CoreFlow Institutional Watchdog with Telegram Alerts

import os
import sys
import time
import logging
import asyncio
import aiohttp
from datetime import datetime

# Configuration
LOG_DIR = "/opt/coreflow/logs"
LOG_FILE = f"{LOG_DIR}/watchdog.log"
RESTART_LIMIT = 5
COOLDOWN_PERIOD = 300  # 5 minutes
TELEGRAM_TIMEOUT = 10  # seconds

# Setup logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CoreFlowWatchdog")

class TelegramNotifier:
    @staticmethod
    async def send(message: str):
        """Send alerts to Telegram"""
        try:
            token = os.getenv("TELEGRAM_TOKEN")
            chat_id = os.getenv("TELEGRAM_CHAT_ID")
            
            if not token or not chat_id:
                logger.warning("Telegram credentials not configured")
                return False

            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": f"🔔 *CoreFlow Alert*\n{message}",
                "parse_mode": "Markdown",
                "disable_notification": False
            }

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TELEGRAM_TIMEOUT)) as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error = await response.text()
                        logger.error(f"Telegram API error: {error}")
                    return response.status == 200

        except Exception as e:
            logger.error(f"Telegram send failed: {str(e)}")
            return False

class ProcessManager:
    def __init__(self):
        self.restart_count = 0
        self.last_restart = 0
        self.process = None
        self.telegram = TelegramNotifier()

    async def start_process(self):
        """Start the CoreFlow main process"""
        try:
            self.process = await asyncio.create_subprocess_exec(
                "python3", "/opt/coreflow/coreflow_main.py",
                stdout=open(f"{LOG_DIR}/coreflow_main.out", "a"),
                stderr=asyncio.subprocess.STDOUT
            )
            self.restart_count += 1
            self.last_restart = time.time()
            
            message = f"✅ Process started (PID: {self.process.pid})"
            logger.info(message)
            await self.telegram.send(message)
            
            return True
        except Exception as e:
            error_msg = f"❌ Failed to start process: {e}"
            logger.error(error_msg)
            await self.telegram.send(error_msg)
            return False

    async def monitor(self):
        """Main monitoring loop"""
        startup_msg = "🚀 CoreFlow Watchdog started"
        logger.info(startup_msg)
        await self.telegram.send(startup_msg)
        
        while True:
            try:
                # Check process status
                if not self.process or self.process.returncode is not None:
                    current_time = time.time()
                    
                    # Check restart limits
                    if (current_time - self.last_restart) < COOLDOWN_PERIOD:
                        if self.restart_count >= RESTART_LIMIT:
                            warning = (
                                f"🚨 Restart limit reached "
                                f"({RESTART_LIMIT} in {COOLDOWN_PERIOD}s)"
                            )
                            logger.warning(warning)
                            await self.telegram.send(warning)
                            await asyncio.sleep(COOLDOWN_PERIOD)
                            continue
                    
                    # Attempt restart
                    if not await self.start_process():
                        await asyncio.sleep(30)
                        continue
                
                # Regular monitoring interval
                await asyncio.sleep(30)
                
            except Exception as e:
                error = f"⚠️ Monitoring error: {e}"
                logger.error(error)
                await self.telegram.send(error)
                await asyncio.sleep(60)

async def main():
    manager = ProcessManager()
    await manager.monitor()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Watchdog stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"🔥 Critical watchdog failure: {e}")
        sys.exit(1)
