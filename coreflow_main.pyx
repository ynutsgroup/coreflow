
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
import asyncio

from core.strategy.valg_engine import FTMO_VALGEngine
from core.config import Config
from core.health_check import HealthMonitor
from utils.telegram_notifier import send_telegram_alert

# --- Load .env ---
load_dotenv("/opt/coreflow/.env")

MAX_RESTARTS = 3
RESTART_DELAY = 5

class ApplicationState:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.shutdown_flag = False
            cls._instance.restart_count = 0
            cls._instance.today_trades = 0
        return cls._instance

class GracefulExiter:
    def __init__(self):
        self.state = ApplicationState()
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame) -> None:
        self.state.shutdown_flag = True
        logging.info(f"📴 Shutdown signal received ({signum})")
        asyncio.run(send_telegram_alert(f"📴 CoreFlow Shutdown (Signal {signum})", "WARNING"))

def setup_logging() -> None:
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
    try:
        monitor = HealthMonitor()
        monitor.start()
        return True
    except Exception as e:
        logging.critical(f"❌ Initialization failed: {str(e)}")
        asyncio.run(send_telegram_alert(f"❌ Initialisierung fehlgeschlagen: {str(e)}", "ERROR"))
        return False

def execute_trade(signal, state):
    if state.today_trades >= 5:
        logging.warning("⚠️ Max daily trades reached")
        asyncio.run(send_telegram_alert("⚠️ Max. Tages-Trades erreicht", "WARNING"))
        return False

    logging.info(f"📈 Executing: {signal['symbol']} {signal['action']} {signal['size']} lot(s)")
    asyncio.run(send_telegram_alert(f"📈 Executing: {signal['symbol']} {signal['action']} {signal['size']} lot(s)", "INFO"))
    state.today_trades += 1
    return True

def main_loop() -> None:
    state = ApplicationState()
    exiter = GracefulExiter()

    try:
        engine = FTMO_VALGEngine(
            account_size=100000,
            max_risk_per_trade=0.01,
            volatility_window=34
        )
        asyncio.run(send_telegram_alert("✅ FTMO Engine gestartet", "INFO"))
    except Exception as e:
        logging.critical(f"❌ Engine init failed: {e}")
        asyncio.run(send_telegram_alert(f"❌ Engine-Start fehlgeschlagen: {e}", "ERROR"))
        sys.exit(1)

    data_path = Path(os.getenv("MARKET_DATA_FILE", "/opt/coreflow/data/market_data.csv"))

    while not state.shutdown_flag:
        try:
            if not data_path.exists():
                logging.warning(f"⚠️ Keine Marktdaten gefunden – Datei fehlt: {data_path}")
                asyncio.run(send_telegram_alert(f"⚠️ Keine Marktdaten gefunden – Datei fehlt: {data_path}", "WARNING"))
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
            asyncio.run(send_telegram_alert(f"❌ Unerwarteter Fehler im Hauptloop: {e}", "ERROR"))
            time.sleep(60)

def main() -> NoReturn:
    setup_logging()
    logger = logging.getLogger("CoreFlow")

    logging.info("🧠 Starting CoreFlow Institutional Trading System")
    asyncio.run(send_telegram_alert("🧠 CoreFlow wird gestartet...", "INFO"))

    if not initialize():
        logger.critical("💥 CoreFlow initialization aborted.")
        sys.exit(1)

    main_loop()

if __name__ == "__main__":
    main()
