#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Institutional Health Monitor for CoreFlow – FTMO & Professional Compliance
"""

import os
import sys
import logging
import threading
import time
import pytz
from datetime import datetime, timedelta, time as dt_time
from typing import Optional, Dict, List
import holidays
import pandas as pd
from dotenv import load_dotenv

load_dotenv("/opt/coreflow/.env")

class HealthMonitor:
    def __init__(
        self,
        max_daily_trades: int = 10,
        max_daily_loss: float = 0.05,
        max_position_size: float = 0.1,
        trading_hours: Dict = None,
        restricted_symbols: List[str] = None
    ):
        # ✅ Logging-Fix: Ausgabe auf stdout, keine Fake-Errors mehr
        self.logger = logging.getLogger("CF.HealthMonitor")
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        stdout_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        stdout_handler.setFormatter(formatter)
        self.logger.addHandler(stdout_handler)

        self.max_daily_trades = max_daily_trades
        self.max_daily_loss = max_daily_loss
        self.max_position_size = max_position_size

        # Trading-Hours aus .env laden oder fallback
        default_start = os.getenv("TRADING_HOURS_FOREX_START", "07:00")
        default_end = os.getenv("TRADING_HOURS_FOREX_END", "22:00")

        default_trading_hours = {
            'forex': {
                'open': dt_time(*map(int, default_start.split(":"))),
                'close': dt_time(*map(int, default_end.split(":"))),
                'timezone': 'UTC'
            }
        }

        self.trading_hours = trading_hours if trading_hours else default_trading_hours

        if 'forex' not in self.trading_hours:
            self.logger.warning("⚠️ Adding fallback trading hours for 'forex'")
            self.trading_hours['forex'] = default_trading_hours['forex']

        self.restricted_symbols = restricted_symbols or []
        self.holidays = holidays.country_holidays('US')

        self.today_trades = 0
        self.daily_pnl = 0.0
        self.current_positions = {}
        self.circuit_breaker = False

        self._shutdown_flag = False
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="HealthMonitor"
        )

    def start(self) -> None:
        self.logger.info("🚀 Starting Institutional Health Monitor")
        self._monitor_thread.start()

    def stop(self) -> None:
        self.logger.info("🛑 Stopping Health Monitor")
        self._shutdown_flag = True
        self._monitor_thread.join()
        self._generate_daily_report()

    def _monitor_loop(self) -> None:
        while not self._shutdown_flag:
            try:
                self._check_trading_conditions()
                self._check_economic_calendar()
                self._check_circuit_breakers()
                time.sleep(30)
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}", exc_info=True)
                time.sleep(60)

    def _check_trading_conditions(self) -> None:
        violations = []

        if self.today_trades >= self.max_daily_trades:
            violations.append("FTMO daily trade limit reached")

        if self.daily_pnl <= -self.max_daily_loss:
            violations.append("FTMO daily loss limit reached")

        if not self._is_market_open():
            violations.append("Outside institutional trading hours")

        for symbol in self.current_positions.keys():
            if symbol in self.restricted_symbols:
                violations.append(f"Restricted symbol traded: {symbol}")

        if violations:
            self._handle_violations(violations)

    def _is_market_open(self) -> bool:
        now = datetime.now(pytz.UTC)

        if now.date() in self.holidays:
            return False

        market_hours = self.trading_hours.get('forex')
        if not market_hours or 'open' not in market_hours or 'close' not in market_hours:
            self.logger.error("❌ Trading hours for 'forex' not defined properly.")
            return False

        tz = pytz.timezone(market_hours.get('timezone', 'UTC'))
        local_now = now.astimezone(tz)
        current_time = local_now.time()

        return market_hours['open'] <= current_time <= market_hours['close']

    def _check_economic_calendar(self) -> None:
        pass  # TODO: API Integration

    def _check_circuit_breakers(self) -> None:
        pass  # TODO: Erweiterung

    def _handle_violations(self, violations: List[str]) -> None:
        for violation in violations:
            msg = f"🚨 Institutional Violation: {violation}"
            self.logger.warning(msg)
            if "daily loss" in violation.lower():
                self.circuit_breaker = True

    def _generate_daily_report(self) -> None:
        report = {
            "date": datetime.now().date().isoformat(),
            "daily_trades": self.today_trades,
            "daily_pnl": self.daily_pnl,
            "rule_violations": self._get_violation_count(),
            "market_hours_compliance": self._get_market_hours_compliance(),
            "position_sizes": self._get_position_sizes()
        }

        report_path = f"/opt/coreflow/reports/{datetime.now().date()}_compliance.json"
        pd.DataFrame([report]).to_json(report_path, indent=2)
        self.logger.info(f"Daily report generated: {report_path}")

    def can_trade(self, symbol: str, size: float) -> bool:
        if self.circuit_breaker:
            return False

        checks = [
            self.today_trades < self.max_daily_trades,
            self.daily_pnl > -self.max_daily_loss,
            size <= self.max_position_size,
            symbol not in self.restricted_symbols,
            self._is_market_open()
        ]

        return all(checks)

    def record_trade(self, symbol: str, pnl: float) -> None:
        self.today_trades += 1
        self.daily_pnl += pnl
        self.current_positions[symbol] = self.current_positions.get(symbol, 0) + 1

    def get_risk_exposure(self) -> Dict:
        return {
            "daily_pnl": self.daily_pnl,
            "position_concentration": self._calculate_concentration(),
            "largest_position": max(self.current_positions.values(), default=0)
        }

    def _calculate_concentration(self) -> float:
        if not self.current_positions:
            return 0.0
        max_pos = max(self.current_positions.values())
        total_pos = sum(self.current_positions.values())
        return max_pos / total_pos if total_pos > 0 else 0.0

    def _get_violation_count(self):
        return 0

    def _get_market_hours_compliance(self):
        return True

    def _get_position_sizes(self):
        return self.current_positions
