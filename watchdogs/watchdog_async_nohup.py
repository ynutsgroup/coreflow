#!/usr/bin/env python3
# CoreFlow Institutional Watchdog – Modular FTMO Edition

import os
import sys
import time
import json
import signal
import logging
import requests
import subprocess
from datetime import datetime
from dotenv import load_dotenv

# --- Konfiguration ---
RESTART_LIMIT = 10
MAX_RESTARTS = 5
COOLDOWN = 600
FTMO_CHECK_INTERVAL = 1800
FTMO_API_BASE = "https://api.ftmo.com/v2"

load_dotenv("/opt/coreflow/.env")

# --- Logging Setup ---
log_dir = os.getenv("LOG_DIR", "/opt/coreflow/logs")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"watchdog_ftmo_{datetime.now().strftime('%Y-%m-%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("CoreFlowWatchdog")

# --- Telegram ---
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "False") == "True"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(msg: str):
    if TELEGRAM_ENABLED and TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=5)
        except Exception as e:
            logger.warning(f"Telegram Error: {e}")

# --- FTMO Monitoring ---
FTMO_MONITOR_ENABLED = os.getenv("FTMO_MONITOR_ENABLED", "False") == "True"

def check_ftmo_connection():
    try:
        r = requests.get(
            f"{FTMO_API_BASE}/ping",
            timeout=10,
            headers={"Authorization": f"Bearer {os.getenv('FTMO_API_KEY')}"}
        )
        if r.status_code == 200:
            return True, "FTMO connection OK"
        return False, f"FTMO API error: {r.status_code}"
    except Exception as e:
        return False, str(e)

def check_ftmo_account_status():
    try:
        r = requests.get(
            f"{FTMO_API_BASE}/account",
            timeout=10,
            headers={"Authorization": f"Bearer {os.getenv('FTMO_API_KEY')}"}
        )
        data = r.json()
        if r.status_code != 200:
            return False, f"API error {r.status_code}: {data.get('message', 'No error message')}"
        if data.get("account_status") == "active":
            return True, "Account active"
        return False, f"Account issue: {data.get('message', 'Unknown')}"
    except Exception as e:
        return False, str(e)

def check_ftmo_limits():
    try:
        r = requests.get(
            f"{FTMO_API_BASE}/account/limits",
            timeout=10,
            headers={"Authorization": f"Bearer {os.getenv('FTMO_API_KEY')}"}
        )
        data = r.json()
        if r.status_code != 200:
            return False, f"API error {r.status_code}: {data.get('message', '')}"
        max_loss = float(data.get('max_daily_loss', 0))
        current_loss = float(data.get('current_daily_loss', 0))
        if max_loss >= current_loss:
            return True, f"Limits OK ({max_loss - current_loss:.2f} remaining)"
        return False, "Daily limit reached"
    except Exception as e:
        return False, str(e)

# --- Restart Count ---
restart_file = os.path.join(log_dir, "watchdog_restart_count.json")

def read_restart_count():
    if not os.path.exists(restart_file):
        return {"count": 0, "last_reset": time.time()}
    try:
        with open(restart_file, "r") as f:
            return json.load(f)
    except:
        return {"count": 0, "last_reset": time.time()}

def write_restart_count(data):
    try:
        with open(restart_file, "w") as f:
            json.dump(data, f)
    except Exception as e:
        logger.error(f"Restart counter write error: {e}")

# --- CoreFlow starten ---
def start_coreflow():
    logger.info("🔁 Restarting coreflow_main.py")
    send_telegram("🔁 CoreFlow restart initiated")
    try:
        return subprocess.Popen(
            ["nohup", "python3", "/opt/coreflow/coreflow_main.py"],
            stdout=open("/opt/coreflow/logs/coreflow_nohup.out", "a"),
            stderr=subprocess.STDOUT,
            preexec_fn=os.setpgrp
        )
    except Exception as e:
        logger.error(f"Failed to start CoreFlow: {e}")
        send_telegram(f"❌ CoreFlow start failed: {e}")
        raise

# --- MAIN ---
def main():
    logger.info("🚀 Starting CoreFlow Watchdog with FTMO Monitoring")
    send_telegram("🚀 CoreFlow Watchdog gestartet")

    if FTMO_MONITOR_ENABLED:
        logger.info("🔍 FTMO Monitoring aktiviert")
        send_telegram("🟢 FTMO Monitoring aktiviert")
        ftmo_checks = [
            (check_ftmo_connection, "Connection"),
            (check_ftmo_account_status, "Account Status"),
            (check_ftmo_limits, "Trading Limits"),
        ]
        for func, name in ftmo_checks:
            success, msg = func()
            if not success:
                logger.error(f"⚠️ FTMO {name} Problem: {msg}")
                send_telegram(f"⚠️ FTMO {name} Alert: {msg}")
            else:
                logger.info(f"✅ FTMO {name}: {msg}")
    else:
        logger.info("🔕 FTMO Monitoring deaktiviert")

    restart_info = read_restart_count()
    if restart_info["count"] >= MAX_RESTARTS and time.time() - restart_info["last_reset"] < COOLDOWN:
        logger.warning("🚫 Max restarts reached – cooldown active")
        send_telegram("🚫 Max restarts – watchdog pausiert")
        return

    if restart_info["count"] > 0:
        logger.info("🧹 Resetting restart counter")
        restart_info = {"count": 0, "last_reset": time.time()}
        write_restart_count(restart_info)

    proc = start_coreflow()
    write_restart_count(restart_info)

    def handle_exit(sig, frame):
        logger.info("🛑 Watchdog stopped manually")
        send_telegram("🛑 Watchdog manuell gestoppt")
        try:
            proc.terminate()
        except Exception as e:
            logger.error(f"Fehler beim Stoppen: {e}")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    last_ftmo_check = time.time()

    while True:
        if proc.poll() is not None:
            restart_info["count"] += 1
            restart_info["last_reset"] = time.time()
            write_restart_count(restart_info)
            logger.warning("⚠️ CoreFlow abgestürzt, Neustart folgt...")
            send_telegram("⚠️ CoreFlow abgestürzt – Neustart läuft")
            time.sleep(5)
            proc = start_coreflow()

        if FTMO_MONITOR_ENABLED and time.time() - last_ftmo_check > FTMO_CHECK_INTERVAL:
            for func, name in ftmo_checks:
                success, msg = func()
                if not success:
                    logger.error(f"⚠️ FTMO {name} Problem: {msg}")
                    send_telegram(f"⚠️ FTMO {name} Alert: {msg}")
            last_ftmo_check = time.time()

        time.sleep(10)

if __name__ == "__main__":
    main()
