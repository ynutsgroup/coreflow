#!/usr/bin/env python3
# CoreFlow Institutional Watchdog – Production Grade

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
from typing import Optional, Dict, List
from dotenv import load_dotenv

# === SETUP ===
load_dotenv("/opt/coreflow/.env")
TARGET_PROCESS = "/opt/coreflow/coreflow_main.py"
LOG_DIR = "/opt/coreflow/logs"
MAX_RESTARTS = 5  # pro Stunde
ALERT_COOLDOWN = 600  # Sekunden pro Fehlertyp
CHECK_INTERVAL = 30  # Sekunden

# GPU Thresholds
MAX_GPU_TEMP = 85
MAX_GPU_LOAD = 95
MAX_GPU_MEM = 90

# === TELEGRAM NOTIFIER ===
class TelegramNotifier:
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.last_alerts: Dict[str, datetime] = {}

    def _should_alert(self, key: str) -> bool:
        now = datetime.now()
        last = self.last_alerts.get(key)
        if not last or (now - last).total_seconds() > ALERT_COOLDOWN:
            self.last_alerts[key] = now
            return True
        return False

    async def send_alert(self, msg: str, key: str = "generic") -> bool:
        if not self._should_alert(key):
            return False
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        data = {
            "chat_id": self.chat_id,
            "text": f"🚨 CoreFlow:\n{msg}",
            "parse_mode": "Markdown"
        }
        try:
            r = requests.post(url, json=data, timeout=10)
            return r.status_code == 200
        except Exception as e:
            logging.error(f"[Telegram] Failed: {e}")
            return False

# === GPU MONITORING ===
class GPUMonitor:
    def __init__(self):
        self.counts = {"temp": 0, "load": 0, "mem": 0}

    async def check(self) -> List[str]:
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            out, _ = await proc.communicate()
            if proc.returncode != 0:
                return []
            temp, load, used, total = map(int, out.decode().strip().split(','))
            mem_pct = used / total * 100
            alerts = []

            if temp > MAX_GPU_TEMP:
                self.counts["temp"] += 1
                if self.counts["temp"] >= 3:
                    alerts.append(f"GPU Temperatur kritisch: {temp}°C > {MAX_GPU_TEMP}°C")
            else:
                self.counts["temp"] = 0

            if load > MAX_GPU_LOAD:
                self.counts["load"] += 1
                if self.counts["load"] >= 3:
                    alerts.append(f"GPU Last hoch: {load}% > {MAX_GPU_LOAD}%")
            else:
                self.counts["load"] = 0

            if mem_pct > MAX_GPU_MEM:
                self.counts["mem"] += 1
                if self.counts["mem"] >= 3:
                    alerts.append(f"GPU Speicher voll: {mem_pct:.1f}% > {MAX_GPU_MEM}%")
            else:
                self.counts["mem"] = 0

            return alerts
        except Exception as e:
            logging.error(f"[GPU] Error: {e}")
            return []

# === PROCESS MONITOR ===
class ProcessWatcher:
    def __init__(self):
        self.restarts: List[datetime] = []
        self.pid: Optional[int] = None

    def is_running(self) -> bool:
        for p in psutil.process_iter(['pid', 'cmdline']):
            try:
                if p.info['cmdline'] and TARGET_PROCESS in ' '.join(p.info['cmdline']):
                    self.pid = p.info['pid']
                    return True
            except:
                continue
        return False

    def restart_allowed(self) -> bool:
        cutoff = datetime.now() - timedelta(hours=1)
        self.restarts = [t for t in self.restarts if t > cutoff]
        return len(self.restarts) < MAX_RESTARTS

    async def restart(self) -> bool:
        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", TARGET_PROCESS,
                cwd=os.path.dirname(TARGET_PROCESS),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self.restarts.append(datetime.now())
            self.pid = proc.pid
            logging.info(f"[Restart] PID: {self.pid}")
            return True
        except Exception as e:
            logging.error(f"[Restart] Failed: {e}")
            return False

# === MAIN LOOP ===
async def main_loop():
    notifier = TelegramNotifier()
    gpu = GPUMonitor()
    watcher = ProcessWatcher()

    await notifier.send_alert("✅ Watchdog gestartet", "startup")
    logging.info("Watchdog gestartet")

    while True:
        try:
            alerts = await gpu.check()
            for msg in alerts:
                await notifier.send_alert(msg, "gpu")

            if not watcher.is_running():
                if watcher.restart_allowed():
                    if await watcher.restart():
                        await notifier.send_alert("🔁 CoreFlow wurde neu gestartet", "restart")
                    else:
                        await notifier.send_alert("❌ Neustart fehlgeschlagen", "fail")
                else:
                    await notifier.send_alert("❌ Max. Neustarts erreicht", "limit")
        except Exception as e:
            logging.error(f"[Loop] {e}")
            await notifier.send_alert(f"❌ Fehler im Watchdog: {e}", "watchdog")

        await asyncio.sleep(CHECK_INTERVAL)

# === ENTRYPOINT ===
def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    logfile = os.path.join(LOG_DIR, "watchdog.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler()
        ]
    )

def main():
    setup_logging()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def stop():
        logging.info("Beende Watchdog...")
        loop.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop)

    try:
        loop.run_until_complete(main_loop())
    finally:
        logging.info("Watchdog beendet")
        loop.close()

if __name__ == "__main__":
    main()
