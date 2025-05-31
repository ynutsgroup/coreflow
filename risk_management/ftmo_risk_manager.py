#!/usr/bin/env python3
"""
FTMO-Compliant Trading System v4.0

Features:
- Vollständiges FTMO-Risikomanagement
- Echtzeit-Signalvalidierung
- Tägliche Limits (Trades, Risiko, Verlust)
- Automatisches Journaling
- Performance-Monitoring
"""

import os
import time
import json
import logging
from pathlib import Path
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

class FTMORiskManager:
    """FTMO-konformes Risikomanagement-System"""
    
    def __init__(
        self,
        max_daily_trades: int = 10,
        max_risk_per_trade: float = 0.01,
        max_daily_loss: float = 0.05,
        starting_balance: float = 100000.0
    ):
        """
        Initialisiert das Risikomanagement mit FTMO-Standardwerten
        
        :param max_daily_trades: Maximale Trades/Tag (FTMO Standard: 10)
        :param max_risk_per_trade: Maximales Risiko/Trade (1% = 0.01)
        :param max_daily_loss: Maximaler Tagesverlust (5% = 0.05)
        :param starting_balance: Startkapital für Risikoberechnungen
        """
        self.max_daily_trades = max_daily_trades
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.starting_balance = starting_balance
        
        self.daily_trade_count = 0
        self.daily_loss = 0.0
        self.daily_profit = 0.0
        self.last_reset_date = datetime.now(timezone.utc).date()
        
    def _check_reset(self):
        """Automatisches Zurücksetzen der Tageszähler bei Tageswechsel"""
        today = datetime.now(timezone.utc).date()
        if today != self.last_reset_date:
            self.daily_trade_count = 0
            self.daily_loss = 0.0
            self.daily_profit = 0.0
            self.last_reset_date = today
            logging.info("FTMO Risikomanagement: Tageszähler zurückgesetzt")

    def validate_trade(
        self,
        risk_amount: float,
        risk_percent: float
    ) -> Tuple[bool, str]:
        """
        Validiert einen Trade gegen FTMO-Regeln
        
        :param risk_amount: Absolutes Risiko in Kontowährung
        :param risk_percent: Risiko in Prozent (0.01 = 1%)
        :return: (is_valid, reason)
        """
        self._check_reset()
        
        if self.daily_trade_count >= self.max_daily_trades:
            return (False, f"Max {self.max_daily_trades} Trades/Tag erreicht")
            
        if risk_percent > self.max_risk_per_trade:
            return (False, f"Risiko {risk_percent*100:.1f}% > Maximum {self.max_risk_per_trade*100:.1f}%")
            
        potential_daily_loss = self.daily_loss + risk_amount
        max_loss_amount = self.starting_balance * self.max_daily_loss
        
        if potential_daily_loss > max_loss_amount:
            return (False, f"Potentieller Tagesverlust {potential_daily_loss:.2f} > Maximum {max_loss_amount:.2f}")
            
        return (True, "Trade validiert")

    def register_trade(
        self,
        pnl_amount: float,
        pnl_percent: float
    ):
        """
        Registriert einen abgeschlossenen Trade
        
        :param pnl_amount: Gewinn/Verlust in Kontowährung
        :param pnl_percent: Gewinn/Verlust in Prozent
        """
        self._check_reset()
        self.daily_trade_count += 1
        
        if pnl_amount < 0:
            self.daily_loss += abs(pnl_amount)
            logging.warning(f"Trade Verlust registriert: {-pnl_amount:.2f}")
        else:
            self.daily_profit += pnl_amount
            
        logging.info(
            f"FTMO Statistik: Trades {self.daily_trade_count}/{self.max_daily_trades} | "
            f"Verlust {self.daily_loss:.2f}/{self.starting_balance*self.max_daily_loss:.2f}"
        )

class FTMOSignalEmitter:
    """FTMO-konformer Signalgenerator mit integriertem Risikomanagement"""
    
    __slots__ = [
        'risk_manager', 'signal_dir', 'journal_dir',
        'strategy_name', 'logger', 'current_balance'
    ]
    
    def __init__(
        self,
        signal_dir: str = '/opt/coreflow/signals',
        journal_dir: str = '/opt/coreflow/journal',
        strategy_name: str = "FTMO_Pro",
        starting_balance: float = 100000.0
    ):
        """
        Initialisiert das FTMO-System
        
        :param starting_balance: Startkapital für Risikoberechnungen
        """
        self.signal_dir = Path(signal_dir)
        self.journal_dir = Path(journal_dir)
        self.strategy_name = strategy_name
        self.current_balance = starting_balance
        
        # Risikomanagement mit FTMO-Standardwerten
        self.risk_manager = FTMORiskManager(
            max_daily_trades=10,
            max_risk_per_trade=0.01,
            max_daily_loss=0.05,
            starting_balance=starting_balance
        )
        
        self._setup_directories()
        self._setup_logging()
        logging.info(f"FTMO System initialisiert | Startkapital: {starting_balance:.2f}")

    def _setup_directories(self):
        """Erstellt benötigte Verzeichnisse"""
        try:
            self.signal_dir.mkdir(parents=True, exist_ok=True)
            self.journal_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.critical(f"Verzeichniserror: {e}")
            raise

    def _setup_logging(self):
        """Konfiguriert das FTMO-Journaling"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.journal_dir/'ftmo_journal.log'),
                logging.StreamHandler()
            ]
        )

    def generate_signal(
        self,
        signal_type: str,
        instrument: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        position_size: float
    ) -> Dict:
        """
        Generiert ein FTMO-validiertes Signal
        
        :return: Signal-Dictionary oder None wenn ungültig
        """
        # Risikoberechnung
        risk_amount = abs(entry_price - stop_loss) * position_size
        risk_percent = risk_amount / self.current_balance
        
        # FTMO-Validierung
        is_valid, reason = self.risk_manager.validate_trade(
            risk_amount, risk_percent
        )
        
        if not is_valid:
            logging.error(f"Signal abgelehnt: {reason}")
            return None
            
        # Signal erstellen
        signal_id = f"FTMO_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        
        signal = {
            'id': signal_id,
            'type': signal_type,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'instrument': instrument,
            'entry': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'size': position_size,
            'risk_amount': risk_amount,
            'risk_percent': risk_percent,
            'reward_ratio': abs(take_profit - entry_price) / abs(entry_price - stop_loss),
            'strategy': self.strategy_name,
            'balance': self.current_balance,
            'validation': {
                'is_valid': is_valid,
                'reason': reason,
                'daily_stats': {
                    'trades': self.risk_manager.daily_trade_count + 1,
                    'max_trades': self.risk_manager.max_daily_trades,
                    'daily_loss': self.risk_manager.daily_loss,
                    'max_daily_loss': self.risk_manager.starting_balance * self.risk_manager.max_daily_loss
                }
            }
        }
        
        # Signal speichern
        try:
            signal_file = self.signal_dir / f"{signal_id}.json"
            with open(signal_file, 'w') as f:
                json.dump(signal, f, indent=2)
                
            logging.info(f"Signal {signal_id} generiert für {instrument}")
            return signal
            
        except Exception as e:
            logging.error(f"Signal konnte nicht gespeichert werden: {e}")
            return None

    def register_trade_outcome(
        self,
        signal_id: str,
        exit_price: float,
        exit_time: datetime,
        commission: float = 0.0
    ):
        """
        Registriert das Ergebnis eines abgeschlossenen Trades
        
        :param signal_id: ID des ursprünglichen Signals
        :param exit_price: Preis bei Trade-Abschluss
        :param exit_time: Zeitpunkt des Abschlusses
        :param commission: Handelsgebühren
        """
        try:
            # Originales Signal laden
            signal_file = self.signal_dir / f"{signal_id}.json"
            with open(signal_file) as f:
                signal = json.load(f)
            
            # PnL berechnen
            entry_price = signal['entry']
            position_size = signal['size']
            
            if signal['type'] == 'long':
                pnl_amount = (exit_price - entry_price) * position_size - commission
            else:  # short
                pnl_amount = (entry_price - exit_price) * position_size - commission
                
            pnl_percent = pnl_amount / self.current_balance
            
            # Kontostand aktualisieren
            self.current_balance += pnl_amount
            
            # Trade registrieren
            self.risk_manager.register_trade(pnl_amount, pnl_percent)
            
            # Journal-Eintrag erstellen
            journal_entry = {
                'signal_id': signal_id,
                'exit_price': exit_price,
                'exit_time': exit_time.isoformat(),
                'pnl_amount': pnl_amount,
                'pnl_percent': pnl_percent,
                'new_balance': self.current_balance,
                'commission': commission,
                'execution_quality': None  # Kann später mit Ausführungsdaten gefüllt werden
            }
            
            journal_file = self.journal_dir / f"exec_{signal_id}.json"
            with open(journal_file, 'w') as f:
                json.dump(journal_entry, f, indent=2)
                
            logging.info(
                f"Trade abgeschlossen | PnL: {pnl_amount:.2f} ({pnl_percent*100:.2f}%) | "
                f"Neuer Kontostand: {self.current_balance:.2f}"
            )
            
        except Exception as e:
            logging.error(f"Trade-Abschluss konnte nicht registriert werden: {e}")

if __name__ == "__main__":
    # Beispielhafte Verwendung
    emitter = FTMOSignalEmitter(starting_balance=50000.0)
    
    # Signal generieren
    signal = emitter.generate_signal(
        signal_type='long',
        instrument='EURUSD',
        entry_price=1.0850,
        stop_loss=1.0820,
        take_profit=1.0920,
        position_size=100000
    )
    
    if signal:
        print(f"Signal erfolgreich generiert: {signal['id']}")
        
        # Simulierten Trade-Abschluss
        emitter.register_trade_outcome(
            signal_id=signal['id'],
            exit_price=1.0900,
            exit_time=datetime.now(timezone.utc)
        )
