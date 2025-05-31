#!/usr/bin/env python3
# CoreFlow Main Execution – Institutional FTMO Edition

import logging
import signal
import sys
import time
import traceback
import pandas as pd
from pathlib import Path
from typing import NoReturn
from dotenv import load_dotenv
import os

from core.strategy.valg_engine import FTMO_VALGEngine
from core.config import Config
from core.health_check import HealthMonitor

# --- Load .env ---
load_dotenv("/opt/coreflow/.env")

MAX_RESTARTS = 3
RESTART_DELAY = 5

class ApplicationState:
    """Globaler Anwendungszustand (Singleton)"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.shutdown_flag = False
            cls._instance.restart_count = 0
            cls._instance.today_trades = 0
        return cls._instance

class GracefulExiter:
    """Fängt SIGINT/SIGTERM für sauberen Shutdown ab"""
    def __init__(self):
        self.state = ApplicationState()
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame) -> None:
        self.state.shutdown_flag = True
        logging.info(f"📴 Shutdown signal received ({signum})")

def setup_logging() -> None:
    """Initialisiert Logging für CoreFlow"""
    log_dir = Path(Config.LOG_DIR)
    log_dir.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "coreflow.log"),
            logging.StreamHandler(),
        ]
    )

def initialize() -> bool:
    """Initialisiert alle Systemkomponenten"""
    try:
        monitor = HealthMonitor()
        monitor.start()
        return True
    except Exception as e:
        logging.critical(f"❌ Initialization failed: {str(e)}")
        return False

def main_loop() -> None:
    """Hauptschleife des Trading-Systems"""
    state = ApplicationState()
    exiter = GracefulExiter()

    try:
        engine = FTMO_VALGEngine(
            account_size=100000,
            max_risk_per_trade=0.01,
            volatility_window=34
        )
    except Exception as e:
        logging.critical(f"❌ Engine init failed: {e}")
        sys.exit(1)

    data_path = Path("/opt/coreflow/market_data.csv")

    while not state.shutdown_flag:
        try:
            if not data_path.exists():
                logging.warning("⚠️ Keine Marktdaten gefunden – Datei fehlt: /opt/coreflow/market_data.csv")
                time.sleep(60)
                continue

            df = pd.read_csv(data_path)
            signals = engine.generate_signals(df)

            if signals['signal'].iloc[-1] != 0:
                signal_data = engine.last_signal
                if execute_trade(signal_data, state):
                    logging.info(f"✅ Executed trade: {signal_data['direction']} | {signal_data['symbol']} | {signal_data['size']} lots")

            time.sleep(60)

        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
            time.sleep(60)

def execute_trade(signal, state):
    """Simulierte Trade-Ausführung mit Limitprüfung"""
    if state.today_trades >= 5:
        logging.warning("⚠️ Max daily trades reached")
        return False
    logging.info(f"📈 Executing: {signal['symbol']} {signal['action']} {signal['size']} lot(s)")
    state.today_trades += 1
    return True

def main() -> NoReturn:
    setup_logging()
    logger = logging.getLogger("CoreFlow")

    logging.info("🧠 Starting CoreFlow Institutional Trading System")

    if not initialize():
        logger.critical("💥 CoreFlow initialization aborted.")
        sys.exit(1)

    main_loop()

if __name__ == "__main__":
    main()
