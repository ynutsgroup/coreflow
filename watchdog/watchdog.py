#!/usr/bin/env python3
# CoreFlow Watchdog - Stable Edition

import os
import sys
import signal
import asyncio
import logging
import psutil
import subprocess
from datetime import datetime, timedelta
from dotenv import load_dotenv
import aiohttp

# === Configuration ===
sys.path.insert(0, '/opt/coreflow')
load_dotenv("/opt/coreflow/.env")

TARGET_PROCESS = os.getenv("WATCHDOG_TARGET", "/opt/coreflow/coreflow_main.py")
LOG_DIR = os.getenv("LOG_DIR", "/opt/coreflow/logs")
RESTART_LIMIT = 5
RESTART_WINDOW = 3600  # 1 hour in seconds
CHECK_INTERVAL = 30
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "false").lower() == "true"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# === State ===
restart_times = []
last_notification = {}

# === Logging Setup ===
def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    logfile = os.path.join(LOG_DIR, f"watchdog_{datetime.now():%Y-%m-%d}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler(sys.stdout)
        ]
    )

# === Process Management ===
def count_recent_restarts():
    """Count restarts within time window"""
    cutoff = datetime.now() - timedelta(seconds=RESTART_WINDOW)
    restart_times[:] = [t for t in restart_times if t > cutoff]
    return len(restart_times)

def is_process_running():
    """Check if target process is actually running"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'status']):
        try:
            if (proc.info['cmdline'] and 
                TARGET_PROCESS in " ".join(proc.info['cmdline']) and
                proc.info['status'] != psutil.STATUS_ZOMBIE):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False

def start_process():
    """Start CoreFlow with resource limits"""
    try:
        # Set resource limits
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))  # Disable core dumps
        resource.setrlimit(resource.RLIMIT_NOFILE, (8192, 8192))
        
        process = subprocess.Popen(
            ["python3", TARGET_PROCESS],
            cwd=os.path.dirname(TARGET_PROCESS),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy(),
            start_new_session=True
        )
        restart_times.append(datetime.now())
        return True
    except Exception as e:
        logging.error(f"Process start failed: {str(e)}", exc_info=True)
        return False

# === Notification System ===
async def send_notification(message, urgent=False):
    """Send notification with cooldown handling"""
    if not TELEGRAM_ENABLED or not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    
    now = datetime.now()
    if not urgent and "last_notification" in globals():
        if (now - last_notification.get("default", now)).total_seconds() < 300:
            return False
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": message,
                    "parse_mode": "HTML",
                    "disable_notification": not urgent
                }
            ) as response:
                if response.status == 200:
                    last_notification["default"] = now
                    return True
    except Exception as e:
        logging.error(f"Notification failed: {str(e)}")
    return False

# === Main Monitoring ===
async def monitor_loop():
    """Main watchdog monitoring logic"""
    setup_logging()
    logging.info("CoreFlow Watchdog started")
    await send_notification("🚀 <b>CoreFlow Watchdog Started</b>", urgent=True)
    
    while True:
        try:
            if not is_process_running():
                restart_count = count_recent_restarts()
                
                if restart_count >= RESTART_LIMIT:
                    msg = (f"⛔ <b>MAX RESTARTS REACHED</b>\n"
                          f"Restarts: {restart_count}/{RESTART_LIMIT}\n"
                          f"Window: {RESTART_WINDOW//3600} hour(s)\n"
                          f"Manual intervention required!")
                    logging.error(msg)
                    await send_notification(msg, urgent=True)
                    await asyncio.sleep(300)  # Longer wait when max restarts hit
                    continue
                
                logging.warning(f"Attempting restart ({restart_count+1}/{RESTART_LIMIT})")
                if start_process():
                    await send_notification(f"✅ <b>Restarted</b> (Attempt {restart_count+1}/{RESTART_LIMIT})")
                else:
                    await send_notification("❌ <b>Restart Failed</b>", urgent=True)
            
            await asyncio.sleep(CHECK_INTERVAL)
            
        except Exception as e:
            logging.error(f"Monitor error: {str(e)}", exc_info=True)
            await send_notification(f"⚠️ <b>Watchdog Error</b>\n{str(e)}", urgent=True)
            await asyncio.sleep(30)

# === Shutdown Handling ===
async def shutdown_handler(signal=None):
    """Handle all shutdown scenarios"""
    signame = signal.name if signal else "manual"
    logging.info(f"Shutdown initiated by {signame}")
    
    # Send stop notification
    await send_notification(f"🛑 <b>Watchdog Stopped</b>\nReason: {signame}", urgent=True)
    
    # Cancel all tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

def handle_signal(signum, frame):
    """Signal handler for all termination cases"""
    asyncio.create_task(shutdown_handler(signal.Signals(signum)))

# === Main Entry Point ===
def main():
    """Application entry point"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Register signal handlers for all termination signals
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGTSTP):
        signal.signal(sig, handle_signal)
    
    try:
        loop.run_until_complete(monitor_loop())
    except asyncio.CancelledError:
        pass  # Expected during shutdown
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
    finally:
        if not loop.is_closed():
            loop.run_until_complete(shutdown_handler())
            loop.close()
        logging.info("Watchdog shutdown complete")

if __name__ == "__main__":
    main()
