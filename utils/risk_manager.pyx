#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FTKO Strict Compliance Risk Manager v4.0

Strikt konform mit FTMO-Regeln:
- Tägliche Verlustlimits (5% Standard)
- Maximale Positionsgröße
- Risikobegrenzung pro Trade (1-2%)
- Trade-Limit (5 Trades/Tag)
- Equity Protection
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import requests
from typing import Optional, Dict, Tuple
from datetime import datetime, timezone

# Load environment variables
load_dotenv('/opt/coreflow/.env')

# FTKO Compliance Constants
FTMO_RULES = {
    'MAX_DAILY_LOSS': 0.05,        # 5% täglicher Verlust
    'MAX_OVERALL_LOSS': 0.10,      # 10% Gesamtverlust
    'MAX_RISK_PER_TRADE': 0.02,    # 2% pro Trade
    'MAX_DAILY_TRADES': 5,         # 5 Trades/Tag
    'MIN_RISK_PER_TRADE': 0.005    # 0.5% Minimum
}

class FTMOComplianceManager:
    def __init__(self, starting_balance: float):
        """Initialize with FTMO rules"""
        self.starting_balance = starting_balance
        self.current_balance = starting_balance
        self.daily_stats = {
            'trades': 0,
            'loss': 0.0,
            'profit': 0.0,
            'last_reset': datetime.now(timezone.utc).date()
        }
        self._setup_logging()
        self._validate_rules()

    def _setup_logging(self):
        """FTMO-konformes Logging"""
        log_dir = '/opt/coreflow/logs/ftmo'
        os.makedirs(log_dir, exist_ok=True)
        
        self.logger = logging.getLogger('FTMO_Compliance')
        self.logger.setLevel(logging.INFO)
        
        handler = RotatingFileHandler(
            f"{log_dir}/compliance.log",
            maxBytes=10*1024*1024,
            backupCount=5
        )
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        ))
        self.logger.addHandler(handler)

    def _validate_rules(self):
        """Validiere FTMO-Regeln"""
        if self.starting_balance <= 0:
            raise ValueError("Startbalance muss positiv sein")
        if FTMO_RULES['MAX_DAILY_LOSS'] >= FTMO_RULES['MAX_OVERALL_LOSS']:
            raise ValueError("Tägliches Limit muss unter Gesamtlimit liegen")

    def _check_daily_reset(self):
        """Automatisches tägliches Reset"""
        today = datetime.now(timezone.utc).date()
        if self.daily_stats['last_reset'] != today:
            self.daily_stats = {
                'trades': 0,
                'loss': 0.0,
                'profit': 0.0,
                'last_reset': today
            }
            self.logger.info("Tägliches Reset durchgeführt")

    def validate_trade(self, risk_percent: float) -> Tuple[bool, str]:
        """Validiere Trade gegen FTMO-Regeln"""
        self._check_daily_reset()
        
        # Regel 1: Tägliches Trade-Limit
        if self.daily_stats['trades'] >= FTMO_RULES['MAX_DAILY_TRADES']:
            return False, f"Max {FTMO_RULES['MAX_DAILY_TRADES']} Trades/Tag erreicht"
        
        # Regel 2: Risiko pro Trade
        if not FTMO_RULES['MIN_RISK_PER_TRADE'] <= risk_percent <= FTMO_RULES['MAX_RISK_PER_TRADE']:
            return False, f"Risiko {risk_percent*100:.2f}% außerhalb der Limits"
        
        # Regel 3: Täglicher Verlust
        daily_loss_limit = self.starting_balance * FTMO_RULES['MAX_DAILY_LOSS']
        if self.daily_stats['loss'] >= daily_loss_limit:
            return False, f"Tägliches Verlustlimit ({daily_loss_limit:.2f}) erreicht"
        
        return True, "FTMO-konformer Trade"

    def register_trade(self, pnl: float, risk_percent: float) -> bool:
        """Registriere einen abgeschlossenen Trade"""
        is_valid, msg = self.validate_trade(risk_percent)
        if not is_valid:
            self.logger.warning(f"Trade abgelehnt: {msg}")
            return False
        
        self.daily_stats['trades'] += 1
        if pnl < 0:
            self.daily_stats['loss'] += abs(pnl)
        else:
            self.daily_stats['profit'] += pnl
        
        self.current_balance += pnl
        self.logger.info(
            f"Trade registriert: PnL {pnl:.2f} | "
            f"Risiko {risk_percent*100:.2f}% | "
            f"Tagesbilanz: {self.daily_stats['profit'] - self.daily_stats['loss']:.2f}"
        )
        return True

    def check_equity_protection(self) -> bool:
        """Überprüfe Equity Protection Rule"""
        overall_loss = self.starting_balance - self.current_balance
        overall_limit = self.starting_balance * FTMO_RULES['MAX_OVERALL_LOSS']
        
        if overall_loss >= overall_limit:
            self.logger.critical(f"EQUITY PROTECTION: Gesamtverlust {overall_loss:.2f} erreicht Limit {overall_limit:.2f}")
            return False
        return True

    def get_risk_parameters(self, symbol: str) -> Dict:
        """FTMO-konforme Risikoparameter für Symbol"""
        # Hier sollte die Symbol-Datenbank abgefragt werden
        return {
            'max_lot_size': self._calculate_max_lot_size(symbol),
            'allowed_leverage': 1:30,
            'min_sl_pips': 20
        }

    def _calculate_max_lot_size(self, symbol: str) -> float:
        """Berechne maximale FTMO-konforme Lot-Größe"""
        # Implementierung der Lot-Berechnung gemäß FTMO-Regeln
        pass

if __name__ == "__main__":
    # Beispielimplementierung
    try:
        manager = FTMOComplianceManager(starting_balance=100000.0)
        
        # Beispiel-Trades
        trades = [
            (1500, 0.01),   # Gewinn-Trade 1% Risiko
            (-2000, 0.02),  # Verlust-Trade 2% Risiko
            (3000, 0.015),  # Gewinn-Trade 1.5% Risiko
            (-1000, 0.01),  # Verlust-Trade 1% Risiko
            (500, 0.005),   # Gewinn-Trade 0.5% Risiko
            (1000, 0.01)    # Sollte abgelehnt werden (Trade-Limit)
        ]
        
        for pnl, risk in trades:
            if not manager.register_trade(pnl, risk):
                print(f"Trade abgelehnt: {pnl} mit {risk*100}% Risiko")
        
        print("Tagesstatistik:", manager.daily_stats)
        
    except Exception as e:
        logging.critical(f"FTKO Compliance Fehler: {str(e)}")
