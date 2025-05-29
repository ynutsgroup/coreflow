#!/usr/bin/env python3
# watchdog.py - CoreFlow Final Version (Linux + GPU Ready)

import os
import time
import signal
import shutil
import logging
import subprocess
from datetime import datetime
from multiprocessing import Queue
from logging.handlers import RotatingFileHandler

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command, CommandObject
from aiogram.utils import executor
from dotenv import load_dotenv
from prometheus_client import start_http_server, Counter

# === Konstanten === 
LOG_DIR = "/var/log/coreflow"
PID_FILE = "/var/run/coreflow.pid"
SCRIPT_PATH = "/opt/coreflow/core/signal_receiver.py"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB

# === Initialisierung === 
load_dotenv("/opt/coreflow/.env")

bot = Bot(token=os.getenv("TELEGRAM_TOKEN"))
dp = Dispatcher()
msg_queue = Queue(maxsize=100)
REQUESTS = Counter('watchdog_commands', 'Received commands')

# === Logger Setup ===
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

# === Signal Handler ===
def handle_shutdown(signum, frame):
    logging.warning(f"Received shutdown signal {signum}")
    stop_coreflow()
    exit(0)

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

# === Prozessmanagement ===
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
        logging.info("CoreFlow process started (PID: %d)", proc.pid)
        msg_queue.put("🚀 CoreFlow erfolgreich gestartet")
    except Exception as e:
        logging.critical(f"Start failed: {str(e)}")
        msg_queue.put(f"💥 Startfehler: {str(e)}")

def stop_coreflow():
    try:
        with open(PID_FILE, "r") as f:
            pid = int(f.read().strip())
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        logging.info("Sent SIGTERM to process group %d", pid)
        msg_queue.put("🛑 CoreFlow gestoppt")
    except FileNotFoundError:
        logging.warning("No PID file found")
        msg_queue.put("⚠️ Kein aktiver Prozess")
    except ProcessLookupError:
        logging.warning("Process already terminated")
        msg_queue.put("⚠️ Prozess war nicht aktiv")

# === System Monitoring ===
def get_system_status():
    status = {
        "load": os.getloadavg(),
        "memory": subprocess.getoutput("free -h"),
        "gpu": subprocess.getoutput("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader") if shutil.which("nvidia-smi") else "N/A"
    }
    return status

# === Telegram Handlers ===
async def is_authorized(message: types.Message) -> bool:
    return str(message.from_user.id) == os.getenv("ADMIN_USER_ID")

async def message_worker():
    while True:
        msg = msg_queue.get()
        try:
            await bot.send_message(os.getenv("TELEGRAM_CHAT_ID"), msg)
        except Exception as e:
            logging.error(f"Message failed: {str(e)}")
        time.sleep(0.1)

@dp.message(Command("restart"))
async def cmd_restart(message: types.Message, command: CommandObject):
    if not await is_authorized(message):
        return

    REQUESTS.inc()
    await message.answer("♻️ Neustart wird durchgeführt...")
    stop_coreflow()
    time.sleep(2)
    start_coreflow()

@dp.message(Command("status"))
async def cmd_status(message: types.Message):
    if not await is_authorized(message):
        return

    REQUESTS.inc()
    status = get_system_status()
    response = (
        "🧠 <b>CoreFlow Status</b>\n"
        f"⏱️ Zeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"📊 Load: {status['load']}\n"
        f"💾 Memory:\n<pre>{status['memory']}</pre>\n"
        f"🎮 GPU: {status['gpu'] or 'N/A'}"
    )
    await message.answer(response, parse_mode="HTML")

# === Main ===
if __name__ == '__main__':
    logging.info("=== CoreFlow Watchdog Starting ===")

    # Starte Message Worker
    import asyncio
    asyncio.get_event_loop().create_task(message_worker())

    # Prometheus Metrics
    start_http_server(8000)

    # Starte CoreFlow
    start_coreflow()

    # Starte Bot
    try:
        executor.start_polling(dp, skip_updates=True)
    except Exception as e:
        logging.critical(f"Bot crashed: {str(e)}")
        msg_queue.put(f"💥 Bot-Absturz: {str(e)}")
        raise
