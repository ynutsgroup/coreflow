#!/usr/bin/env python3
# CoreFlow Institutional Watchdog – Linux/FTMO Edition (v2.1)

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

# Configuration
TARGET_PROCESS = "/opt/coreflow/coreflow_main.py"
LOG_DIR = "/opt/coreflow/logs"
MAX_RESTARTS = 5  # Max restarts per hour
ALERT_COOLDOWN = 600  # Seconds between duplicate alerts

# GPU Thresholds
MAX_GPU_TEMP = 85  # °C
MAX_GPU_LOAD = 95   # %
MAX_GPU_MEM = 90    # %

class TelegramNotifier:
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.session = requests.Session()
        self.last_alerts: Dict[str, datetime] = {}

    def _should_alert(self, alert_key: str) -> bool:
        """Prevent alert flooding"""
        if alert_key not in self.last_alerts:
            return True
        return (datetime.now() - self.last_alerts[alert_key]).total_seconds() > ALERT_COOLDOWN

    async def send_alert(self, message: str, alert_key: str = "generic") -> bool:
        """Send alert with built-in retry logic"""
        if not self._should_alert(alert_key):
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": f"🚨 CoreFlow Alert:\n{message}",
            "parse_mode": "Markdown"
        }

        try:
            # First try - async with timeout
            try:
                async with self.session.post(url, json=payload, timeout=10) as resp:
                    if resp.status_code == 200:
                        self.last_alerts[alert_key] = datetime.now()
                        return True
            except Exception as e:
                logging.warning(f"Telegram async attempt failed: {str(e)}")

            # Fallback - sync request
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                self.last_alerts[alert_key] = datetime.now()
                return True

            logging.error(f"Telegram API error: {resp.status_code} - {resp.text}")
            return False

        except Exception as e:
            logging.error(f"Telegram send failed: {str(e)}")
            return False

class GPUMonitor:
    def __init__(self):
        self.alert_counts = {"temp": 0, "load": 0, "mem": 0}

    async def check_status(self) -> List[str]:
        """Check GPU metrics and return alerts if thresholds exceeded"""
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                return []

            temp, load, mem_used, mem_total = map(int, stdout.decode().strip().split(','))
            mem_pct = (mem_used / mem_total) * 100

            alerts = []
            if temp > MAX_GPU_TEMP:
                self.alert_counts["temp"] += 1
                if self.alert_counts["temp"] >= 3:
                    alerts.append(f"GPU Temperature Critical: {temp}°C (Max {MAX_GPU_TEMP}°C)")
            else:
                self.alert_counts["temp"] = 0

            if load > MAX_GPU_LOAD:
                self.alert_counts["load"] += 1
                if self.alert_counts["load"] >= 3:
                    alerts.append(f"GPU Load High: {load}% (Max {MAX_GPU_LOAD}%)")
            else:
                self.alert_counts["load"] = 0

            if mem_pct > MAX_GPU_MEM:
                self.alert_counts["mem"] += 1
                if self.alert_counts["mem"] >= 3:
                    alerts.append(f"GPU Memory High: {mem_pct:.1f}% (Max {MAX_GPU_MEM}%)")
            else:
                self.alert_counts["mem"] = 0

            return alerts

        except Exception as e:
            logging.error(f"GPU check failed: {str(e)}")
            return []

class ProcessWatcher:
    def __init__(self):
        self.restart_times: List[datetime] = []
        self.process_pid: Optional[int] = None
        self.failure_count = 0

    def is_running(self) -> bool:
        """Check if target process is running"""
        for proc in psutil.process_iter(['pid', 'cmdline']):
            try:
                if proc.info['cmdline'] and TARGET_PROCESS in ' '.join(proc.info['cmdline']):
                    self.process_pid = proc.info['pid']
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False

    def should_restart(self) -> bool:
        """Check if process can be restarted"""
        # Remove old restart times
        cutoff = datetime.now() - timedelta(hours=1)
        self.restart_times = [t for t in self.restart_times if t > cutoff]
        return len(self.restart_times) < MAX_RESTARTS

    async def restart_process(self) -> bool:
        """Attempt to restart the target process"""
        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", TARGET_PROCESS,
                cwd=os.path.dirname(TARGET_PROCESS),
                env=os.environ,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self.restart_times.append(datetime.now())
            self.process_pid = proc.pid
            self.failure_count = 0
            return True
        except Exception as e:
            self.failure_count += 1
            logging.error(f"Process restart failed: {str(e)}")
            return False

async def main_loop():
    """Main watchdog monitoring loop"""
    # Initialize components
    notifier = TelegramNotifier()
    gpu_monitor = GPUMonitor()
    process_watcher = ProcessWatcher()

    # Send startup notification
    await notifier.send_alert("Watchdog service started", "startup")

    while True:
        try:
            # Check GPU status
            gpu_alerts = await gpu_monitor.check_status()
            for alert in gpu_alerts:
                await notifier.send_alert(alert, "gpu_warning")

            # Check process status
            if not process_watcher.is_running():
                if process_watcher.should_restart():
                    if await process_watcher.restart_process():
                        await notifier.send_alert(
                            f"Process restarted (PID: {process_watcher.process_pid})", 
                            "process_restart"
                        )
                    else:
                        await notifier.send_alert(
                            f"Restart failed ({process_watcher.failure_count} attempts)", 
                            "restart_failed"
                        )
                else:
                    await notifier.send_alert(
                        f"Max restarts reached ({MAX_RESTARTS}/hour)", 
                        "restart_limit"
                    )

            # Sleep before next check
            await asyncio.sleep(30)

        except Exception as e:
            logging.error(f"Main loop error: {str(e)}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    # Configure logging
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(LOG_DIR, "watchdog.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handle shutdown signals
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def shutdown():
        """Clean shutdown handler"""
        await notifier.send_alert("Watchdog service stopping", "shutdown")
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        [task.cancel() for task in tasks]
        await asyncio.gather(*tasks, return_exceptions=True)
        loop.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))

    try:
        loop.run_until_complete(main_loop())
    finally:
        loop.close()
