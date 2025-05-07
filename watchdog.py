# watchdog.py – CoreFlow Selbstüberwachung mit Restart-Logik

#!/usr/bin/env python3

import sys
import os
import subprocess
import time
import psutil
import logging
import requests

# Projektpfad hinzufügen
sys.path.append("/opt/coreflow")

from config.settings import config
from core.auto_pauser import AutoPauser
from utils.notifier import send_telegram_message

COREFLOW_SCRIPT = "/opt/coreflow/scripts/coreflow_main.py"
RESTART_DELAY = 5
MAX_RESTARTS = 5
LOG_FILE = "/var/log/coreflow_watchdog.log"

def kill_orphan_processes():
    """Beendet verwaiste coreflow_main.py Prozesse."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if proc.info['cmdline'] and 'coreflow_main.py' in ' '.join(proc.info['cmdline']):
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                pass

def run_watchdog():
    send_telegram_message("🛡️ CoreFlow Watchdog gestartet")
    restart_count = 0

    with open(LOG_FILE, "a") as log_file:
        while restart_count < MAX_RESTARTS:
            try:
                kill_orphan_processes()
                log_file.write(f"\n[{time.ctime()}] Starte CoreFlow...\n")
                process = subprocess.Popen(
                    ["python3", COREFLOW_SCRIPT],
                    stdout=log_file,
                    stderr=log_file
                )
                exit_code = process.wait()
                log_file.write(f"[{time.ctime()}] Beendet mit Code {exit_code}\n")
                send_telegram_message(f"⚠️ CoreFlow beendet (Code {exit_code})")

                restart_count = 0 if exit_code == 0 else restart_count + 1
            except Exception as e:
                log_file.write(f"[{time.ctime()}] FEHLER: {str(e)}\n")
                restart_count += 1
                send_telegram_message(f"❌ Watchdog Fehler: {str(e)}")

            time.sleep(RESTART_DELAY)

        send_telegram_message("🔴 Max. Neustarts erreicht – Watchdog pausiert 1 Stunde.")
        time.sleep(3600)

if __name__ == "__main__":
    run_watchdog()
