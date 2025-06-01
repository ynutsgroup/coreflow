#!/usr/bin/env python3
# CoreFlow Main Execution – Institutional FTMO Edition (Stable v2.5)

import os
import sys
import time
import signal
import traceback
import logging
import asyncio
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

sys.path.insert(0, '/opt/coreflow')

from core.config import Config
from core.strategy.valg_engine import FTMO_VALGEngine
from core.health_check import HealthMonitor
from utils.telegram_notifier import send_telegram_alert

# --- FTMO Konstante ---
FTMO_MAX_DAILY_LOSS = 0.05
FTMO_MAX_OVERALL_LOSS = 0.10
FTMO_MIN_TRADING_DAYS = 5
FTMO_MAX_DAILY_TRADES = 10
FTMO_ACCOUNT_SIZE = 100000

# --- Risk Manager ---
class FTMO_RiskManager:
    def __init__(self, account_size):
        self.account_size = account_size
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.today_trades = 0
        self.trading_days = 0
        self.last_trade_day = None

    def update_pnl(self, pnl):
        today = datetime.utcnow().date()
        if self.last_trade_day != today:
            if self.last_trade_day is not None:
                self.trading_days += 1
            self.last_trade_day = today
            self.daily_pnl = 0.0
            self.today_trades = 0
        self.daily_pnl += pnl
        self.total_pnl += pnl
        self.today_trades += 1

    def check_limits(self):
        if self.daily_pnl < -FTMO_MAX_DAILY_LOSS * self.account_size:
            return False
        if self.total_pnl < -FTMO_MAX_OVERALL_LOSS * self.account_size:
            return False
        if self.today_trades >= FTMO_MAX_DAILY_TRADES:
            return False
        return True

    def get_risk_per_trade(self):
        return 0.005 if self.daily_pnl < 0 else 0.01

# --- Application State ---
class ApplicationState:
    def __init__(self):
        self.shutdown_flag = False
        self.risk_manager = None

# --- Logging Setup ---
def setup_logging():
    log_dir = Path(Config.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "coreflow.log"),
            logging.StreamHandler()
        ]
    )

# --- Initialization ---
async def initialize(state):
    try:
        HealthMonitor().start()
        state.risk_manager = FTMO_RiskManager(FTMO_ACCOUNT_SIZE)
        logging.info("✅ FTMO System initialization completed")
        return True
    except Exception as e:
        logging.critical(f"Initialization failed: {e}")
        await send_telegram_alert(f"❌ Init-Fehler: {e}", "ERROR")
        return False

# --- Execute Trade ---
async def execute_trade(state, engine, signal):
    if not state.risk_manager.check_limits():
        logging.warning("🚫 Trade geblockt: Risikolimit erreicht")
        await send_telegram_alert("🚫 Trade geblockt: Risikolimit erreicht", "WARNING")
        return

    risk_pct = state.risk_manager.get_risk_per_trade()
    try:
        result = await engine.execute_trade(
            signal=signal,
            risk_percent=risk_pct,
            account_size=FTMO_ACCOUNT_SIZE
        )
        state.risk_manager.update_pnl(result.get("pnl", 0.0))
        logging.info(f"✅ Trade ausgeführt: {result}")
        await send_telegram_alert(f"✅ Trade: {result}", "INFO")
    except Exception as e:
        logging.error(f"Trade-Fehler: {e}")
        await send_telegram_alert(f"⚠️ Trade-Fehler: {e}", "ERROR")

# --- Main Loop ---
async def main_loop(state, engine):
    data_path = Path(os.getenv("MARKET_DATA_FILE", "/opt/coreflow/data/market_data.csv"))
    last_check = 0

    while not state.shutdown_flag:
        try:
            if not state.risk_manager.check_limits():
                state.shutdown_flag = True
                logging.critical("🛑 Limit verletzt – CoreFlow stoppt")
                await send_telegram_alert("🛑 Limit verletzt – CoreFlow stoppt", "ERROR")
                break

            if time.time() - last_check >= 60:
                if not data_path.exists():
                    logging.warning("📁 market_data.csv nicht gefunden")
                    await send_telegram_alert("⚠️ market_data.csv fehlt!", "WARNING")
                    await asyncio.sleep(30)
                    continue

                try:
                    df = pd.read_csv(data_path)
                    signals = engine.generate_signals(df)
                    for signal in signals:
                        await execute_trade(state, engine, signal)
                    last_check = time.time()
                except Exception as e:
                    logging.error(f"📉 Fehler bei Datenverarbeitung: {e}")
                    await send_telegram_alert(f"⚠️ Datenfehler: {e}", "ERROR")
                    await asyncio.sleep(30)

            await asyncio.sleep(1)

        except Exception as e:
            logging.error(f"💥 Hauptloop-Fehler: {e}\n{traceback.format_exc()}")
            await send_telegram_alert(f"⚠️ Hauptloop-Fehler: {e}", "ERROR")
            await asyncio.sleep(10)

# --- Async Main ---
async def async_main():
    setup_logging()
    load_dotenv("/opt/coreflow/.env")
    logging.info("🧠 Starting CoreFlow Institutional Trading System")
    await send_telegram_alert("🧠 CoreFlow wird gestartet...", "INFO")

    state = ApplicationState()
    engine = FTMO_VALGEngine(
        account_size=FTMO_ACCOUNT_SIZE,
        max_risk_per_trade=0.01,
        volatility_window=34
    )

    if not await initialize(state):
        logging.critical("❌ Init fehlgeschlagen")
        sys.exit(1)

    await main_loop(state, engine)

# --- Start ---
if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("⛔ Manuell beendet")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Abbruch: {e}")
        sys.exit(1)
