#!/usr/bin/env python3
# CoreFlow Institutional Watchdog – Linux/FTMO Edition
# 🔁 Auto-Restart | 📩 Telegram Alerts | 🧠 Prozessüberwachung | ✅ VENV-kompatibel

import os
import time
import signal
import logging
import subprocess
from datetime import datetime, timedelta

from dotenv import load_dotenv

# === 🔐 ENV-Laden ===
load_dotenv('/opt/coreflow/.env')

# === ⚙️ Konfiguration ===
COREFLOW_CMD = ["/opt/coreflow/.venv/bin/python3", "/opt/coreflow/coreflow_main.py"]
RESTART_COOLDOWN = 10        # Sekunden zwischen Neustarts
MAX_RESTARTS_PER_HOUR = 12   # Anti-Endlosschleife
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

LOG_PATH = "/opt/coreflow/logs/watchdog_async.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# === 📜 Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)

# === 📩 Telegram-Funktion ===
def send_telegram(msg: str):
    try:
        import requests
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
        requests.post(url, data=data, timeout=5)
    except Exception as e:
        logging.warning(f"Telegram-Fehler: {e}")

# === 🧠 Hauptfunktion ===
def run_watchdog():
    logging.info("🎯 Watchdog gestartet")
    send_telegram("🟢 CoreFlow Watchdog gestartet")

    restart_times = []

    while True:
        logging.info("🔁 Starte coreflow_main.py ...")
        process = subprocess.Popen(COREFLOW_CMD)

        try:
            process.wait()
        except Exception as e:
            logging.error(f"Fehler im CoreFlow-Prozess: {e}")
        finally:
            process.terminate()

        logging.warning("⚠️ CoreFlow-Prozess abgestürzt, Neustart folgt...")

        send_telegram("⚠️ CoreFlow abgestürzt – Neustart wird durchgeführt")

        # 🕓 Neustart-Limitierung
        now = datetime.now()
        restart_times = [rt for rt in restart_times if now - rt < timedelta(hours=1)]
        restart_times.append(now)

        if len(restart_times) > MAX_RESTARTS_PER_HOUR:
            logging.critical("🛑 Zu viele Neustarts – 1h Pause")
            send_telegram("🛑 CoreFlow: Zu viele Neustarts – pausiert 1h")
            time.sleep(3600)
            restart_times.clear()
        else:
            time.sleep(RESTART_COOLDOWN)

# === 🚀 Startpunkt ===
if __name__ == "__main__":
    try:
        run_watchdog()
    except KeyboardInterrupt:
        logging.info("⏹️ Watchdog manuell beendet")
        send_telegram("⏹️ CoreFlow Watchdog manuell beendet")
