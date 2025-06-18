#!/usr/bin/env python3
# /opt/coreflow/core/health_check.py

import threading
import time
import logging
import psutil
import requests
import asyncio
from datetime import datetime
from utils.telegram_notifier import send_telegram_alert


class InstitutionalHealthMonitor:
    """Institutional-grade trading system health monitor (FTMO-ready)"""

    def __init__(self, interval=60, mt5_api_url="http://localhost:5000/mt5/stats"):
        self.interval = interval
        self.mt5_api_url = mt5_api_url
        self.running = False
        self.thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.performance_stats = {
            'start_time': datetime.now(),
            'checks': 0,
            'failures': 0
        }

        # Configure institutional-grade logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler('/var/log/coreflow/health.log'),
                logging.StreamHandler()
            ]
        )
        logging.info("📡 HealthMonitor Logging aktiv")

    def start(self):
        """Start the monitoring process"""
        self.running = True
        self.thread.start()
        logging.info("🩺 Institutional HealthMonitor gestartet")

    def stop(self):
        """Stop the monitoring process"""
        self.running = False
        self.thread.join()
        logging.info("🛑 HealthMonitor gestoppt")

    def check_system_resources(self):
        """Check critical system resources"""
        return {
            'cpu': psutil.cpu_percent(interval=1) < 90,
            'memory': psutil.virtual_memory().percent < 85,
            'disk': psutil.disk_usage('/').percent < 90
        }

    def check_trading_conditions(self):
        """Verify trading environment conditions"""
        try:
            response = requests.get(self.mt5_api_url, timeout=5)
            data = response.json()
            return {
                'mt5_connected': data.get('connected', False),
                'account_equity': data.get('equity', 0),
                'active_orders': data.get('orders', 0)
            }
        except Exception as e:
            logging.error(f"MT5 API Fehler: {str(e)}")
            asyncio.run(self._alert_async(f"❌ MT5 API Fehler: {e}"))
            return {
                'mt5_connected': False,
                'error': str(e)
            }

    def risk_assessment(self):
        """FTMO-style risk evaluation"""
        return {
            'max_drawdown': self.calculate_drawdown(),
            'daily_loss_limit': self.check_daily_loss(),
            'position_sizing': self.verify_position_sizes()
        }

    def calculate_drawdown(self):
        """Placeholder for drawdown logic"""
        return 0.0

    def check_daily_loss(self):
        """Placeholder for daily loss logic"""
        return True

    def verify_position_sizes(self):
        """Placeholder for position sizing logic"""
        return True

    def monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self.performance_stats['checks'] += 1
                system = self.check_system_resources()
                if not all(system.values()):
                    self.performance_stats['failures'] += 1
                    logging.warning(f"Systemüberlastung erkannt: {system}")

                mt5 = self.check_trading_conditions()
                if not mt5.get("mt5_connected", False):
                    logging.error("🚨 MT5-Verbindung verloren")

                risk = self.risk_assessment()
                if risk.get("max_drawdown", 0) > 5:
                    logging.critical(f"⚠️ Drawdown zu hoch: {risk['max_drawdown']}%")

                if self.performance_stats['checks'] % 10 == 0:
                    uptime = datetime.now() - self.performance_stats['start_time']
                    logging.info(
                        f"✅ Statusbericht | Uptime: {uptime} | Checks: {self.performance_stats['checks']} | Fehler: {self.performance_stats['failures']}"
                    )

            except Exception as e:
                logging.error(f"💥 Monitoring Fehler: {str(e)}")
                try:
                    asyncio.run(self._alert_async(f"💥 Monitoring Fehler: {e}"))
                except Exception as alert_error:
                    logging.error(f"⚠️ Telegram-Fehler: {alert_error}")

            time.sleep(self.interval)

    def get_performance_stats(self):
        uptime = datetime.now() - self.performance_stats['start_time']
        checks = self.performance_stats['checks']
        failures = self.performance_stats['failures']
        return {
            **self.performance_stats,
            'uptime': str(uptime),
            'success_rate': 1 - (failures / checks) if checks > 0 else 1.0
        }

    async def _alert_async(self, message):
        """Async helper for sending Telegram alerts from threads"""
        await send_telegram_alert(message)


if __name__ == "__main__":
    monitor = InstitutionalHealthMonitor(interval=30)
    monitor.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop()
        print("Performance:", monitor.get_performance_stats())
