#!/usr/bin/env python3
import logging
import signal
import sys
import time
import traceback
import pandas as pd
from pathlib import Path
from typing import NoReturn
from core.strategy.valg_engine import FTMO_VALGEngine
from core.config import Config
from core.health_check import HealthMonitor

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

    def exit_gracefully(self, signum, frame) -> None:
        self.state.shutdown_flag = True
        logging.info(f"Received shutdown signal {signum}")

def setup_logging() -> None:
    """Setup logging for CoreFlow"""
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
    """Initialize system components"""
    try:
        HealthMonitor().start()
        return True
    except Exception as e:
        logging.critical(f"Initialization failed: {str(e)}")
        return False

def main_loop() -> None:
    """Main application loop"""
    state = ApplicationState()
    exiter = GracefulExiter()

    engine = FTMO_VALGEngine(account_size=100000, max_risk_per_trade=0.01, volatility_window=34)

    while not state.shutdown_flag:
        try:
            df = pd.read_csv("market_data.csv")  # Example of reading market data
            signals = engine.generate_signals(df)

            if signals['signal'].iloc[-1] != 0:
                signal = engine.last_signal
                if execute_trade(signal, state):
                    logging.info(f"Executed trade: {signal['direction']} {signal['size']} lots")

            time.sleep(60)  # Check every minute
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            time.sleep(60)

def execute_trade(signal, state):
    """Executes trade if conditions are met"""
    if state.today_trades >= 5:  # Example daily trade limit check
        logging.warning("Max daily trades reached")
        return False
    logging.info(f"Executing trade: {signal['symbol']} {signal['action']}")
    state.today_trades += 1
    return True

def main() -> NoReturn:
    """Main function"""
    setup_logging()
    logger = logging.getLogger(__name__)

    if not initialize():
        sys.exit(1)

    main_loop()

if __name__ == "__main__":
    main()
