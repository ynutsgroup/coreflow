#!/usr/bin/env python3
# coreflow_watchdog.py - Production-Grade Monitoring Service

import os
import time
import signal
import shutil
import logging
import subprocess
from datetime import datetime
from multiprocessing import Queue
from logging.handlers import RotatingFileHandler

import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandObject
from aiogram.types import Message
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv
from prometheus_client import start_http_server, Counter

# ==================== CONSTANTS ====================
LOG_DIR = "/var/log/coreflow"
PID_FILE = "/run/coreflow.pid"
SCRIPT_PATH = "/opt/coreflow/core/signal_receiver.py"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
PROMETHEUS_PORT = 8000

# ==================== INITIALIZATION ====================
load_dotenv("/opt/coreflow/.env")

# Initialize Telegram Bot
bot = Bot(token=os.getenv("TELEGRAM_TOKEN"), default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()
msg_queue = Queue(maxsize=100)
REQUESTS = Counter('watchdog_commands', 'Received commands')

# ==================== LOGGING SETUP ====================
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        RotatingFileHandler(
            f"{LOG_DIR}/watchdog.log",
            maxBytes=MAX_LOG_SIZE,
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== SIGNAL HANDLERS ====================
def handle_shutdown(signum, frame):
    logger.warning(f"Received shutdown signal {signum}")
    stop_coreflow()
    exit(0)

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

# ==================== PROCESS MANAGEMENT ====================
def write_pid(pid: int):
    with open(PID_FILE, "w") as f:
        f.write(str(pid))

def start_coreflow():
    try:
        proc = subprocess.Popen(
            ["python3", SCRIPT_PATH],
            stdout=open(f"{LOG_DIR}/signal_stdout.log", "a"),
            stderr=open(f"{LOG_DIR}/signal_stderr.log", "a"),
            preexec_fn=os.setsid
        )
        write_pid(proc.pid)
        logger.info(f"CoreFlow process started (PID: {proc.pid})")
        msg_queue.put("🚀 CoreFlow successfully started")
    except Exception as e:
        logger.critical(f"Start failed: {str(e)}")
        msg_queue.put(f"💥 Startup error: {str(e)}")

def stop_coreflow():
    try:
        with open(PID_FILE, "r") as f:
            pid = int(f.read().strip())
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        logger.info(f"Sent SIGTERM to process group {pid}")
        msg_queue.put("🛑 CoreFlow stopped")
    except FileNotFoundError:
        logger.warning("No PID file found")
        msg_queue.put("⚠️ No active process")
    except ProcessLookupError:
        logger.warning("Process already terminated")
        msg_queue.put("⚠️ Process was not active")

# ==================== SYSTEM MONITORING ====================
def get_system_status():
    return {
        "load": os.getloadavg(),
        "memory": subprocess.getoutput("free -h"),
        "gpu": get_gpu_status(),
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def get_gpu_status():
    if shutil.which("nvidia-smi"):
        try:
            return subprocess.getoutput(
                "nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader"
            )
        except:
            return "N/A (Error)"
    return "N/A (No NVIDIA GPU)"

# ==================== TELEGRAM HANDLERS ====================
async def is_authorized(message: Message) -> bool:
    return str(message.from_user.id) == os.getenv("ADMIN_USER_ID")

async def message_worker():
    while True:
        msg = msg_queue.get()
        try:
            await bot.send_message(os.getenv("TELEGRAM_CHAT_ID"), msg)
        except Exception as e:
            logger.error(f"Message failed: {str(e)}")
        await asyncio.sleep(0.1)

@dp.message(Command("restart"))
async def cmd_restart(message: Message, command: CommandObject):
    if not await is_authorized(message):
        return
    REQUESTS.inc()
    await message.answer("♻️ Performing restart...")
    stop_coreflow()
    await asyncio.sleep(2)
    start_coreflow()

@dp.message(Command("status"))
async def cmd_status(message: Message):
    if not await is_authorized(message):
        return
    REQUESTS.inc()
    status = get_system_status()
    response = (
        "🧠 <b>CoreFlow Status Report</b>\n"
        f"⏱️ Time: {status['timestamp']}\n"
        f"📊 Load: {status['load']}\n"
        f"💾 Memory:\n<pre>{status['memory']}</pre>\n"
        f"🎮 GPU:\n<pre>{status['gpu']}</pre>"
    )
    await message.answer(response, parse_mode=ParseMode.HTML)

# ==================== MAIN EXECUTION ====================
if __name__ == '__main__':
    logger.info("=== Starting CoreFlow Watchdog ===")
    loop = asyncio.get_event_loop()
    loop.create_task(message_worker())
    start_http_server(PROMETHEUS_PORT)
    logger.info(f"Prometheus metrics server started on port {PROMETHEUS_PORT}")
    start_coreflow()
    try:
        logger.info("Starting Telegram bot polling")
        loop.run_until_complete(dp.start_polling(bot))
    except Exception as e:
        logger.critical(f"Bot crashed: {str(e)}")
        msg_queue.put(f"💥 Bot crash: {str(e)}")
        raise
