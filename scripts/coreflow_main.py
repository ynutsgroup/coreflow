#!/usr/bin/env python3
import sys
import os
sys.path.append("/opt/coreflow")
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from utils.notifier import send_telegram_message
from config.settings import config
from core.auto_pauser import AutoPauser
# from core.market_scanner import SpreadScanner  # Nur wenn vorhanden
# from core.ai.ftmo_ai_trader import FTMOAgent, TradingConfig  # Optional

logger = logging.getLogger(__name__)

class CoreFlowCommander:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        # self.scanner = SpreadScanner()
        # self.agent = FTMOAgent(TradingConfig())
        self.auto_pauser = AutoPauser(idle_threshold_minutes=30)
        self.start_time = time.time()

    def run(self):
        send_telegram_message("🚀 CoreFlow Commander gestartet!")
        logger.info("CoreFlow Commander gestartet")

        while True:
            start_cycle = time.time()
            # spreads = self.scanner.get_spreads()
            # decisions = self.agent.decide_trades(spreads)

            # Dummy-Aktivität
            self.auto_pauser.update_activity()

            if self.auto_pauser.should_pause():
                sleep_time = 60
            else:
                sleep_time = config.SCAN_INTERVAL

            uptime = int(time.time() - self.start_time)
            if uptime % config.HEARTBEAT_INTERVAL < sleep_time:
                send_telegram_message(f"💓 CoreFlow Uptime: {uptime} Sekunden", important=False)

            elapsed = time.time() - start_cycle
            time.sleep(max(0, sleep_time - elapsed))

    def shutdown(self):
        logger.info("CoreFlow ordnungsgemäß beendet")
        send_telegram_message("🛑 CoreFlow Commander manuell gestoppt", important=True)
        self.executor.shutdown()
        logger.info("Dienste heruntergefahren")

def main():
    try:
        commander = CoreFlowCommander()
        commander.run()
    except KeyboardInterrupt:
        commander.shutdown()
    except Exception as e:
        logger.critical(f"Kritischer Fehler: {str(e)}", exc_info=True)
        send_telegram_message(f"💥 Systemabsturz: {str(e)}", important=True)

if __name__ == "__main__":
    main()
