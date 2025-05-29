#!/usr/bin/env python3
"""CoreFlow Main Application Entry Point"""

import logging
import signal
import sys
import time
import traceback
from typing import NoReturn, Optional
from pathlib import Path
from core.config import Config
from core.signal_receiver import listen
from core.health_check import HealthMonitor

# Constants
MAX_RESTARTS = 3
RESTART_DELAY = 5

class ApplicationState:
    """Globaler Anwendungszustand"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.shutdown_flag = False
            cls._instance.restart_count = 0
        return cls._instance

class GracefulExiter:
    """Handles graceful shutdown signals"""
    def __init__(self):
        self.state = ApplicationState()
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        signal.signal(signal.SIGHUP, self.reload_config)  # Für Konfig-Neuladen

    def exit_gracefully(self, signum, frame) -> None:
        """Trigger graceful shutdown"""
        self.state.shutdown_flag = True
        logging.getLogger(__name__).info(
            f"Received shutdown signal {signum}", extra={"signal": signum}
        )

    def reload_config(self, signum, frame) -> None:
        """Handle config reload"""
        logging.getLogger(__name__).info("Reloading configuration...")
        Config.reload()
        logging.getLogger(__name__).info("Configuration reloaded")

def setup_logging() -> None:
    """Konfiguriert erweitertes Logging-System"""
    log_dir = Path(Config.LOG_DIR)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    logging.basicConfig(
        level=Config.LOG_LEVEL,
        format='%(asctime)s.%(msecs)03d [%(process)d] %(name)-25s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(
                filename=log_dir / 'coreflow.log',
                encoding='utf-8',
                delay=False
            ),
            logging.StreamHandler(),
        ]
    )
    logging.captureWarnings(True)
    logging.getLogger("redis").setLevel(logging.WARNING)

def initialize() -> bool:
    """Initialisiert Systemkomponenten"""
    try:
        # Health Check Subsystem
        HealthMonitor().start()
        
        # Weitere Initialisierungen hier
        return True
    except Exception as e:
        logging.critical(f"Initialization failed: {str(e)}")
        return False

def main_loop() -> None:
    """Hauptverarbeitungsschleife"""
    state = ApplicationState()
    exiter = GracefulExiter()
    logger = logging.getLogger(__name__)
    
    while not state.shutdown_flag:
        try:
            listen()  # Haupt-Verarbeitungsfunktion
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error: {str(e)}")
            state.restart_count += 1
            if state.restart_count > MAX_RESTARTS:
                logger.critical("Max restarts reached, shutting down")
                raise
            time.sleep(RESTART_DELAY)
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise

def main() -> NoReturn:
    """Hauptanwendungslogik"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🚀 CoreFlow v2.0 starting (PID: %d)", os.getpid())
        logger.info("Running with config:\n%s", Config.summary())
        
        if not initialize():
            sys.exit(1)

        main_loop()
            
    except KeyboardInterrupt:
        logger.info("🛑 Graceful shutdown initiated")
    except Exception as e:
        logger.critical(
            "❌ Critical failure: %s\n%s", 
            str(e), 
            traceback.format_exc()
        )
        sys.exit(1)
    finally:
        cleanup()
        logger.info("🔌 CoreFlow shutdown complete")
        sys.exit(0)

def cleanup() -> None:
    """Aufräumarbeiten beim Herunterfahren"""
    HealthMonitor().stop()
    # Weitere Cleanup-Tasks hier

if __name__ == "__main__":
    import os
    import redis  # Für spezifische Exception-Handling
    main()
