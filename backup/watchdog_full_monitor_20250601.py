#!/usr/bin/env python3
# CoreFlow Institutional Watchdog – Linux/FTMO Edition (Updated)

import os
import sys
import time
import signal
import asyncio
import logging
import psutil
import subprocess
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Fix import path issues
sys.path.insert(0, '/opt/coreflow')

# Import Telegram notifier with fallback
try:
    from utils.telegram_notifier import send_telegram_alert as send_telegram_message
except ImportError as e:
    sys.path.insert(0, '/opt')
    try:
        from coreflow.utils.telegram_notifier import send_telegram_alert as send_telegram_message
    except ImportError:
        logging.error("Failed to import Telegram notifier from both paths")
        raise

# Load configuration
load_dotenv("/opt/coreflow/.env")

# Constants
TARGET_PROCESS = os.getenv("WATCHDOG_TARGET", "/opt/coreflow/coreflow_main.py")
LOG_DIR = os.getenv("LOG_DIR", "/opt/coreflow/logs")
RESTART_LIMIT = 5
RESTART_WINDOW = 3600  # Seconds
COOLDOWN_PERIOD = 600  # Anti-spam in seconds

# State variables
restart_times = []
last_error_sent = {}

def setup_logging():
    """Configure logging system"""
    today = datetime.now().strftime("%Y-%m-%d")
    logfile = os.path.join(LOG_DIR, f"watchdog_{today}.log")
    os.makedirs(LOG_DIR, exist_ok=True)
    
    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Add console logging
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

def count_recent_restarts():
    """Count restarts within the time window"""
    cutoff = datetime.now() - timedelta(seconds=RESTART_WINDOW)
    return len([t for t in restart_times if t > cutoff])

def should_throttle(error_key):
    """Check if we should throttle messages for this error type"""
    now = datetime.now()
    if error_key in last_error_sent:
        if (now - last_error_sent[error_key]).total_seconds() < COOLDOWN_PERIOD:
            logging.debug(f"Throttling active for {error_key}")
            return True
    last_error_sent[error_key] = now
    return False

def is_process_running():
    """Check if target process is running"""
    for proc in psutil.process_iter(['cmdline']):
        try:
            if proc.info['cmdline'] and TARGET_PROCESS in " ".join(proc.info['cmdline']):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return False

def start_coreflow():
    """Start the CoreFlow process"""
    logging.info("🔁 CoreFlow-Prozess wird neu gestartet (subprocess)...")
    try:
        subprocess.Popen(
            ["python3", TARGET_PROCESS],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd="/opt/coreflow/",
            env=os.environ.copy()
        )
        restart_times.append(datetime.now())
        return True
    except Exception as e:
        logging.error(f"❌ Fehler beim Starten von CoreFlow: {e}", exc_info=True)
        return False

async def send_telegram_with_fallback(message):
    """Send Telegram message with async/sync fallback"""
    try:
        # Try async version first
        try:
            result = await asyncio.wait_for(
                send_telegram_message(message),
                timeout=10.0
            )
            if result:
                return True
            logging.warning("Telegram async returned False")
        except asyncio.TimeoutError:
            logging.warning("Telegram async timeout")
        except Exception as e:
            logging.warning(f"Telegram async error: {str(e)}")

        # Fallback to sync version
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{os.getenv('TELEGRAM_BOT_TOKEN')}/sendMessage",
                data={
                    "chat_id": os.getenv("TELEGRAM_CHAT_ID"),
                    "text": message
                },
                timeout=10
            )
            if resp.status_code == 200:
                return True
            logging.warning(f"Telegram sync bad status: {resp.status_code}")
        except Exception as e:
            logging.error(f"Telegram sync error: {str(e)}")

        return False
    except Exception as e:
        logging.error(f"Unexpected error in send_telegram_with_fallback: {str(e)}")
        return False

async def monitor_loop():
    """Main monitoring loop"""
    setup_logging()
    logging.info("📡 CoreFlow Watchdog gestartet.")
    
    # Initial test message
    if not await send_telegram_with_fallback("📡 CoreFlow Watchdog gestartet."):
        logging.error("Initial Telegram message failed")

    while True:
        try:
            if not is_process_running():
                if count_recent_restarts() >= RESTART_LIMIT:
                    msg = f"❌ Max. Neustartanzahl erreicht ({RESTART_LIMIT}/{RESTART_WINDOW}s)."
                    if not should_throttle("restart_limit"):
                        logging.warning(msg)
                        await send_telegram_with_fallback(msg)
                    await asyncio.sleep(60)
                    continue
                
                if start_coreflow():
                    await send_telegram_with_fallback("✅ CoreFlow wurde neu gestartet.")
                else:
                    await send_telegram_with_fallback("⚠️ CoreFlow Neustart fehlgeschlagen!")
            
            await asyncio.sleep(30)
        except Exception as e:
            msg = f"❌ Watchdog-Fehler: {str(e)}"
            logging.error(msg, exc_info=True)
            if not should_throttle("watchdog_error"):
                await send_telegram_with_fallback(msg)
            await asyncio.sleep(30)

async def shutdown():
    """Clean shutdown handler"""
    logging.info("🛑 Watchdog wird beendet...")
    await send_telegram_with_fallback("🛑 CoreFlow Watchdog beendet.")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    await asyncio.gather(*tasks, return_exceptions=True)

def main():
    """Main entry point"""
    # Create new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Register signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda sig=sig: asyncio.create_task(shutdown()))
    
    try:
        loop.run_until_complete(monitor_loop())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"Main loop error: {str(e)}", exc_info=True)
    finally:
        loop.close()
        logging.info("Watchdog shutdown complete")

if __name__ == "__main__":
    main()
