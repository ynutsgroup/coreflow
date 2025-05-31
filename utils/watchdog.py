#!/usr/bin/env python3
"""
CoreFlow Institutional Watchdog – Linux/FTMO Edition
"""

import os
import sys
import time
import signal
import logging
import logging.handlers
import subprocess
import asyncio
import threading
import psutil
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from telegram import Bot
from telegram.error import TelegramError

# --- Load .env ---
load_dotenv("/opt/coreflow/.env")

# --- Configuration ---
WATCHDOG_TARGET = os.getenv("WATCHDOG_TARGET", "/opt/coreflow/coreflow_main.py")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", 5))
RESTART_LIMIT = int(os.getenv("RESTART_LIMIT", 3))
RESTART_WINDOW = int(os.getenv("RESTART_WINDOW", 1800))
COOLDOWN_PERIOD = int(os.getenv("COOLDOWN_PERIOD", 600))
MAX_CPU_USAGE = float(os.getenv("MAX_CPU_USAGE", 80.0))
MAX_MEMORY_USAGE = float(os.getenv("MAX_MEMORY_USAGE", 75.0))
MIN_DISK_GB = float(os.getenv("MIN_DISK_GB", 5.0))
LOG_DIR = Path(os.getenv("LOG_DIR", "/opt/coreflow/logs"))

class Watchdog:
    def __init__(self):
        self.process = None
        self.restart_timestamps = []
        self.bot = Bot(token=TELEGRAM_TOKEN)
        self.logger = self.setup_logging()

    def setup_logging(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("CF.InstitutionalWatchdog")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")

        file_handler = logging.handlers.RotatingFileHandler(
            LOG_DIR / "watchdog.log", maxBytes=5*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger

    async def send_telegram(self, message, level="INFO"):
        try:
            text = f"*[{level}]*\n`{datetime.now().isoformat()}`\n```\n{message}\n```"
            await asyncio.get_event_loop().run_in_executor(None, self.bot.send_message,
                TELEGRAM_CHAT_ID, text, "MarkdownV2")
        except TelegramError as e:
            self.logger.error(f"Telegram send failed: {e}")

    def launch_process(self):
        try:
            self.process = subprocess.Popen(
                [sys.executable, WATCHDOG_TARGET],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid
            )
            self.logger.info(f"Started: PID {self.process.pid}")
            return True
        except Exception as e:
            self.logger.error(f"Launch failed: {e}")
            return False

    def is_process_running(self):
        return self.process and self.process.poll() is None

    def terminate_process(self):
        if self.process and self.is_process_running():
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=10)
            except Exception:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            finally:
                self.process = None

    def monitor_output(self):
        def read_stream(stream, is_err):
            while self.process and self.is_process_running():
                line = stream.readline()
                if line:
                    level = logging.ERROR if is_err else logging.INFO
                    self.logger.log(level, line.strip())

        threading.Thread(target=read_stream, args=(self.process.stdout, False), daemon=True).start()
        threading.Thread(target=read_stream, args=(self.process.stderr, True), daemon=True).start()

    def restart_needed(self):
        now = time.time()
        self.restart_timestamps = [t for t in self.restart_timestamps if now - t < RESTART_WINDOW]
        if len(self.restart_timestamps) >= RESTART_LIMIT:
            return True
        self.restart_timestamps.append(now)
        return False

    async def check_system_health(self):
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').free / (1024**3)

        issues = []
        if cpu > MAX_CPU_USAGE:
            issues.append(f"High CPU: {cpu:.1f}%")
        if mem > MAX_MEMORY_USAGE:
            issues.append(f"High Memory: {mem:.1f}%")
        if disk < MIN_DISK_GB:
            issues.append(f"Low Disk Space: {disk:.1f} GB")

        if issues:
            await self.send_telegram("\n".join(issues), level="WARNING")

    async def run(self):
        if not Path(WATCHDOG_TARGET).exists():
            self.logger.error(f"Target {WATCHDOG_TARGET} not found.")
            await self.send_telegram(f"Target not found: {WATCHDOG_TARGET}", "CRITICAL")
            return

        self.launch_process()
        self.monitor_output()
        await self.send_telegram(f"Watchdog started.\nTarget: {WATCHDOG_TARGET}")

        while True:
            await self.check_system_health()

            if not self.is_process_running():
                self.logger.warning("Process stopped.")
                await self.send_telegram("Process stopped.", "WARNING")
                self.terminate_process()

                if self.restart_needed():
                    msg = f"Restart limit reached. Cooling down {COOLDOWN_PERIOD}s."
                    self.logger.warning(msg)
                    await self.send_telegram(msg, "WARNING")
                    await asyncio.sleep(COOLDOWN_PERIOD)
                    self.restart_timestamps.clear()

                self.launch_process()
                self.monitor_output()
                await self.send_telegram("Process restarted.")

            await asyncio.sleep(CHECK_INTERVAL)

def main():
    watchdog = Watchdog()
    try:
        asyncio.run(watchdog.run())
    except KeyboardInterrupt:
        watchdog.logger.info("Watchdog terminated.")

if __name__ == "__main__":
    main()
