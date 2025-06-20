#!/usr/bin/env python3
import os
import sys
import time
import signal
import logging
import asyncio
import subprocess
from datetime import datetime

class CoreFlowWatchdog:
    def __init__(self):
        self._setup_logging()
        self._load_config()
        self.process = None
        self.running = True
        self.restart_count = 0

    def _setup_logging(self):
        """Configure logging to both console and file"""
        self.logger = logging.getLogger("CoreFlowWatchdog")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        log_file = f"/opt/coreflow/logs/watchdog_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _load_config(self):
        """Load configuration from environment variables"""
        self.config = {
            "main_script": os.getenv("MAIN_SCRIPT", "/opt/coreflow/main.py"),
            "check_interval": int(os.getenv("CHECK_INTERVAL", "30")),
            "restart_limit": int(os.getenv("RESTART_LIMIT", "5")),
            "cooldown_period": int(os.getenv("COOLDOWN_PERIOD", "300"))
        }

    async def _start_process(self):
        """Start the main process with proper error handling"""
        try:
            if not os.path.exists(self.config["main_script"]):
                raise FileNotFoundError(f"Main script not found at {self.config['main_script']}")
            
            self.process = subprocess.Popen(
                ["python3", self.config["main_script"]],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                universal_newlines=True
            )
            self.logger.info(f"Started main process (PID: {self.process.pid})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start process: {str(e)}")
            return False

    async def _monitor(self):
        """Main monitoring loop"""
        while self.running:
            if not self.process or self.process.poll() is not None:
                if self.restart_count >= self.config["restart_limit"]:
                    self.logger.error(f"Restart limit ({self.config['restart_limit']}) reached!")
                    break
                
                self.restart_count += 1
                self.logger.warning(f"Attempting restart #{self.restart_count}")
                
                if not await self._start_process():
                    await asyncio.sleep(self.config["cooldown_period"])
                    continue
            
            await asyncio.sleep(self.config["check_interval"])

    def _shutdown(self, signum, frame):
        """Clean shutdown handler"""
        self.logger.info(f"Received signal {signum} - Shutting down")
        self.running = False
        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        sys.exit(0)

async def main():
    watchdog = CoreFlowWatchdog()
    signal.signal(signal.SIGTERM, watchdog._shutdown)
    signal.signal(signal.SIGINT, watchdog._shutdown)
    await watchdog._monitor()

if __name__ == "__main__":
    asyncio.run(main())
