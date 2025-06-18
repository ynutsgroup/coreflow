#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FTMO Compliance Module – CoreFlow v1.0
Checks and tracks all FTMO-relevant trading rules.
"""

from datetime import datetime
import threading
import logging

class FTMOComplianceManager:
    """Enhanced FTMO Compliance System"""
    def __init__(self, starting_balance: float):
        self.starting_balance = starting_balance
        self.current_balance = starting_balance
        self.daily_stats = {
            'trades': 0,
            'loss': 0.0,
            'profit': 0.0,
            'last_reset': datetime.now().date()
        }
        self.lock = threading.Lock()
        self._setup_rules()

    def _setup_rules(self):
        """Initialize FTMO rules"""
        self.rules = {
            'MAX_DAILY_LOSS': 0.05,        # 5% daily loss
            'MAX_OVERALL_LOSS': 0.10,      # 10% total loss
            'MAX_RISK_PER_TRADE': 0.02,    # 2% per trade
            'MAX_DAILY_TRADES': 5,         # 5 trades/day
            'MIN_RISK_PER_TRADE': 0.005    # 0.5% minimum
        }

    def _check_daily_reset(self):
        """Auto-reset at midnight"""
        today = datetime.now().date()
        if self.daily_stats['last_reset'] != today:
            with self.lock:
                self.daily_stats = {
                    'trades': 0,
                    'loss': 0.0,
                    'profit': 0.0,
                    'last_reset': today
                }
            logging.info("Daily stats reset")

    def validate_trade(self, risk_percent: float):
        """Validate a trade against FTMO rules"""
        self._check_daily_reset()

        if self.daily_stats['trades'] >= self.rules['MAX_DAILY_TRADES']:
            return False, f"Max daily trades ({self.rules['MAX_DAILY_TRADES']}) reached"

        if not (self.rules['MIN_RISK_PER_TRADE'] <= risk_percent <= self.rules['MAX_RISK_PER_TRADE']):
            return False, f"Risk {risk_percent:.2%} outside allowed range"

        daily_loss_limit = self.starting_balance * self.rules['MAX_DAILY_LOSS']
        if self.daily_stats['loss'] >= daily_loss_limit:
            return False, f"Daily loss limit ({daily_loss_limit:.2f}) reached"

        return True, "Trade approved"

    def register_trade(self, pnl: float, risk_percent: float) -> bool:
        """Register a trade and update stats"""
        with self.lock:
            valid, reason = self.validate_trade(risk_percent)
            if not valid:
                logging.warning(f"Trade rejected: {reason}")
                return False

            self.daily_stats['trades'] += 1
            if pnl < 0:
                self.daily_stats['loss'] += abs(pnl)
            else:
                self.daily_stats['profit'] += pnl
            self.current_balance += pnl

            logging.info(f"Trade registered | PnL: {pnl:.2f} | Risk: {risk_percent:.2%}")
            return True

    def check_equity_protection(self) -> bool:
        """Check overall equity protection"""
        loss = self.starting_balance - self.current_balance
        max_loss = self.starting_balance * self.rules['MAX_OVERALL_LOSS']
        return loss < max_loss

# ─────────────────────────────────────────────────────────
# TESTMODE
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🔍 FTMO Compliance Testlauf – Standalone\n")

    ftmo = FTMOComplianceManager(starting_balance=100000)

    # 1. Gültiger Trade
    valid, reason = ftmo.validate_trade(0.01)
    print(f"✅ Validierung 1: {valid} – {reason}")

    success = ftmo.register_trade(pnl=-300.0, risk_percent=0.01)
    print(f"📊 Registrierung 1 (Verlust): {success}")

    # 2. Gewinntrade
    success = ftmo.register_trade(pnl=700.0, risk_percent=0.01)
    print(f"📈 Registrierung 2 (Gewinn): {success}")

    # 3. Risiko zu hoch
    valid, reason = ftmo.validate_trade(0.05)
    print(f"🚫 Validierung 3 (zu viel Risiko): {valid} – {reason}")

    # 4. Equity-Schutz?
    if ftmo.check_equity_protection():
        print("🟢 Equity innerhalb der Limits")
    else:
        print("🔴 Equity Limit überschritten")
