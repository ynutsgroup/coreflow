#!/usr/bin/env python3
"""
FTMO Risk Manager Module v3.4

Final stable version with:
- No external dependencies (pytz removed)
- All syntax errors fixed
- Modern Python practices
"""

import os
import json
import logging
import logging.handlers
import threading
from datetime import datetime, timezone
from typing import Tuple, Dict, Any, Optional

class FTMORiskManager:
    """Advanced FTMO Risk Manager with persistent state and enhanced features."""
    
    _instance = None
    _lock = threading.Lock()
    STATE_FILE = "/var/lib/coreflow/ftmo_risk_state.json"
    
    def __new__(cls, *args, **kwargs):
        """Thread-safe singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        max_daily_trades: Optional[int] = None,
        max_risk_per_trade: Optional[float] = None,
        max_daily_loss: Optional[float] = None,
        starting_balance: Optional[float] = None
    ):
        """Initialize risk manager with flexible configuration."""
        with self._lock:
            if self._initialized:
                return

            # Default configuration
            self.config = {
                'max_daily_trades': 5,
                'max_risk_per_trade': 0.02,
                'max_daily_loss': 0.05,
                'starting_balance': 100000.0,
                'instrument_rules': {
                    'crypto': {'max_risk_multiplier': 0.8},
                    'forex': {'max_risk_multiplier': 1.0}
                }
            }

            # Update with provided config
            if config:
                self.config.update(config)
            if max_daily_trades is not None:
                self.config['max_daily_trades'] = max_daily_trades
            if max_risk_per_trade is not None:
                self.config['max_risk_per_trade'] = max_risk_per_trade
            if max_daily_loss is not None:
                self.config['max_daily_loss'] = max_daily_loss
            if starting_balance is not None:
                self.config['starting_balance'] = starting_balance

            # Validate configuration
            self._validate_config()
            
            # Initialize logging
            self._setup_logging()
            
            # Initialize state
            self.daily_trade_count = 0
            self.daily_loss = 0.0
            self.daily_profit = 0.0
            self.last_reset_date = datetime.now(timezone.utc).date()
            
            # Performance tracking
            self.performance_stats = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'max_win': 0,
                'max_loss': 0,
                'total_profit': 0,
                'total_loss': 0
            }
            
            # Load saved state if exists
            self.load_state()
            self._initialized = True
            self.logger.info("FTMO Risk Manager initialized (UTC timezone)")

    def _validate_config(self):
        """Validate configuration parameters."""
        if not isinstance(self.config['max_daily_trades'], int) or self.config['max_daily_trades'] <= 0:
            raise ValueError("max_daily_trades must be positive integer")
        if not isinstance(self.config['max_risk_per_trade'], (int, float)) or self.config['max_risk_per_trade'] <= 0:
            raise ValueError("max_risk_per_trade must be positive number")
        if not isinstance(self.config['max_daily_loss'], (int, float)) or self.config['max_daily_loss'] <= 0:
            raise ValueError("max_daily_loss must be positive number")
        if not isinstance(self.config['starting_balance'], (int, float)) or self.config['starting_balance'] <= 0:
            raise ValueError("starting_balance must be positive number")

    def _setup_logging(self):
        """Configure logging system."""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        
        # File handler with rotation
        log_dir = "/var/log/coreflow"
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            f"{log_dir}/ftmo_risk.log",
            maxBytes=10*1024*1024,
            backupCount=5
        )
        fh.setFormatter(formatter)
        
        self.logger = logging.getLogger('FTMORiskManager')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

    def reset_daily_counts(self):
        """Reset daily counters."""
        with self._lock:
            self.daily_trade_count = 0
            self.daily_loss = 0.0
            self.daily_profit = 0.0
            self.last_reset_date = datetime.now(timezone.utc).date()
            self.save_state()
            self.logger.info("Reset daily counts")

    def _auto_reset(self):
        """Auto-reset counters at UTC midnight."""
        today = datetime.now(timezone.utc).date()
        if self.last_reset_date != today:
            with self._lock:
                if self.last_reset_date != today:
                    self.reset_daily_counts()

    def save_state(self):
        """Save current state to file."""
        state = {
            'daily_trade_count': self.daily_trade_count,
            'daily_loss': self.daily_loss,
            'daily_profit': self.daily_profit,
            'last_reset_date': self.last_reset_date.isoformat(),
            'performance_stats': self.performance_stats,
            'config': self.config
        }
        
        try:
            os.makedirs(os.path.dirname(self.STATE_FILE), exist_ok=True)
            with open(self.STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")

    def load_state(self):
        """Load state from file."""
        try:
            if os.path.exists(self.STATE_FILE):
                with open(self.STATE_FILE, 'r') as f:
                    state = json.load(f)
                
                self.daily_trade_count = state.get('daily_trade_count', 0)
                self.daily_loss = state.get('daily_loss', 0.0)
                self.daily_profit = state.get('daily_profit', 0.0)
                self.last_reset_date = datetime.fromisoformat(
                    state.get('last_reset_date', datetime.now(timezone.utc).date().isoformat())
                ).date()
                self.performance_stats = state.get('performance_stats', {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'max_win': 0,
                    'max_loss': 0,
                    'total_profit': 0,
                    'total_loss': 0
                })
                self._auto_reset()
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
            self.reset_daily_counts()

    def validate_trade(self, trade_value: float, risk_percentage: float,
                     instrument: Optional[str] = None) -> Tuple[bool, str]:
        """Validate trade against risk rules."""
        if not isinstance(trade_value, (int, float)):
            raise TypeError("trade_value must be numeric")
        if not isinstance(risk_percentage, (int, float)) or abs(risk_percentage) > 1:
            raise ValueError("risk_percentage must be between -1 and 1")
            
        self._auto_reset()
        
        with self._lock:
            # Check trade limit
            if self.daily_trade_count >= self.config['max_daily_trades']:
                return False, f"Max {self.config['max_daily_trades']} trades reached"
                
            # Check risk per trade
            max_risk = self.config['max_risk_per_trade']
            if instrument and instrument.lower() in self.config['instrument_rules']:
                max_risk *= self.config['instrument_rules'][instrument.lower()]['max_risk_multiplier']
                
            if abs(risk_percentage) > max_risk:
                return False, f"Risk {abs(risk_percentage)*100:.2f}% exceeds max {max_risk*100:.2f}%"
                
            # Check daily loss
            if trade_value < 0:
                potential_loss = self.daily_loss + trade_value
                max_loss = self.config['max_daily_loss'] * self.config['starting_balance']
                
                if abs(potential_loss) > max_loss:
                    return False, f"Potential loss {abs(potential_loss):.2f} exceeds max {max_loss:.2f}"
                    
            return True, "Trade valid"

    def register_trade(self, trade_value: float, risk_percentage: float,
                     instrument: Optional[str] = None) -> None:
        """Register completed trade."""
        if not isinstance(trade_value, (int, float)):
            raise TypeError("trade_value must be numeric")
        if not isinstance(risk_percentage, (int, float)) or abs(risk_percentage) > 1:
            raise ValueError("risk_percentage must be between -1 and 1")
            
        self._auto_reset()
        
        with self._lock:
            # Update counts
            self.daily_trade_count += 1
            if trade_value < 0:
                self.daily_loss += trade_value
            else:
                self.daily_profit += trade_value
                
            # Update performance stats
            self.performance_stats['total_trades'] += 1
            if trade_value >= 0:
                self.performance_stats['winning_trades'] += 1
                self.performance_stats['max_win'] = max(self.performance_stats['max_win'], trade_value)
                self.performance_stats['total_profit'] += trade_value
            else:
                self.performance_stats['losing_trades'] += 1
                self.performance_stats['max_loss'] = min(self.performance_stats['max_loss'], trade_value)
                self.performance_stats['total_loss'] += trade_value
                
            # Log trade
            self.logger.info(
                f"Trade #{self.daily_trade_count}: Value={trade_value:.2f} "
                f"Risk={abs(risk_percentage)*100:.2f}% Instrument={instrument or 'N/A'}"
            )
            self.save_state()

    def get_daily_stats(self) -> Dict[str, Any]:
        """Get current daily statistics."""
        self._auto_reset()
        
        with self._lock:
            total_trades = self.performance_stats['total_trades']
            winning_trades = self.performance_stats['winning_trades']
            losing_trades = self.performance_stats['losing_trades']
            total_profit = self.performance_stats['total_profit']
            total_loss = self.performance_stats['total_loss']
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            profit_factor = abs(total_profit / abs(total_loss)) if total_loss < 0 else float('inf')
            
            return {
                'trade_count': self.daily_trade_count,
                'max_trades': self.config['max_daily_trades'],
                'daily_loss': self.daily_loss,
                'daily_profit': self.daily_profit,
                'net_pnl': self.daily_profit + self.daily_loss,
                'max_daily_loss': self.config['max_daily_loss'] * self.config['starting_balance'],
                'remaining_daily_loss': max(
                    0,
                    self.config['max_daily_loss'] * self.config['starting_balance'] + self.daily_loss
                ),
                'last_reset': self.last_reset_date.isoformat(),
                'performance': {
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades
                }
            }

    def force_reset(self) -> None:
        """Force reset of daily counts."""
        with self._lock:
            self.reset_daily_counts()
            self.logger.warning("Forced reset performed")

def get_ftmo_risk_manager() -> FTMORiskManager:
    """Get singleton risk manager instance."""
    return FTMORiskManager()
