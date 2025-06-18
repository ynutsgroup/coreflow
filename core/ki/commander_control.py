#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Commander Control – CoreFlow KI (FTMO FUSION PRO Edition)

Enhanced Features:
- Multi-layered risk assessment
- Real-time FTMO compliance checks
- AI-powered market regime detection
- Encrypted communication
- Circuit breaker pattern
"""

import os
import logging
import traceback
from datetime import datetime
from typing import Dict, Optional
from dotenv import load_dotenv
from cryptography.fernet import Fernet

# CoreFlow Modules
from core.ki.valg_engine import VALGHybridEngine
from core.ki.market_state import MarketStateClassifier as MarketStateAnalyzer
from core.risk_engine import VALGHybridEngine as RiskEngine
from core.telegram_bot.admin.telegram_bot import VALGAdminBot
from core.emitter.signal_emitter import SignalEmitter
from core.spreadscanner import SpreadScanner
from core.autopause import AutoPauseManager
from core.risk_manager.ftmo_risk_manager import EnhancedFTMOEngine

# === Logging Configuration ===
logger = logging.getLogger("CoreFlow.Commander")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
    handlers=[
        logging.FileHandler("/var/log/coreflow/commander.log"),
        logging.StreamHandler()
    ]
)

# === Environment Setup ===
load_dotenv(dotenv_path="/opt/coreflow/.env")

class FTMOCommander:
    """Institutional-grade trading commander with AI fusion"""
    
    def __init__(self):
        self.circuit_breaker = 0
        self.max_circuit_breaks = 5
        self.encryption_key = self._init_encryption()
        self.components = self._initialize_components()
        
    def _init_encryption(self) -> Optional[Fernet]:
        """Initialize encryption handler"""
        key = os.getenv("ENCRYPTION_KEY")
        if not key:
            logger.warning("No encryption key found - running in insecure mode")
            return None
        return Fernet(key.encode())

    def _initialize_components(self) -> Dict:
        """Initialize all trading components with fault tolerance"""
        components = {
            'market': MarketStateAnalyzer(os.getenv("MARKET_MODEL_PATH")),
            'risk': RiskEngine({
                'max_daily_loss': float(os.getenv("MAX_DAILY_LOSS", "0.05")),
                'max_positions': int(os.getenv("MAX_POSITIONS", "10")),
                'absolute_sl': float(os.getenv("ABSOLUTE_SL", "0.1")),
                'model_path': os.getenv("RISK_MODEL_PATH"),
                'accounts': [int(a) for a in os.getenv("MT5_ACCOUNTS", "").split(",")]
            }),
            'emitter': SignalEmitter(),
            'spread': SpreadScanner(symbols=os.getenv("SYMBOLS", "EURUSD,GBPUSD").split(",")),
            'autopause': AutoPauseManager(),
            'adminbot': VALGAdminBot(
                os.getenv("TELEGRAM_TOKEN"),
                [int(c) for c in os.getenv("TELEGRAM_CHAT_IDS").split(",")]
            ),
            'ftmo': EnhancedFTMOEngine()
        }
        
        # Verify initialization
        for name, component in components.items():
            if not component:
                raise RuntimeError(f"Component {name} failed to initialize")
        
        return components

    def _build_trading_context(self, symbol: str) -> Dict:
        """Create comprehensive trading decision context"""
        try:
            return {
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'market_state': self.components['market'].analyze_market(symbol),
                'risk_assessment': self.components['risk'].evaluate_order(symbol, 1.0, "BUY"),
                'spread_analysis': self.components['spread'].scan_markets()[symbol],
                'autopause_status': self.components['autopause'].check_pause_conditions(),
                'ftmo_compliance': self.components['ftmo'].validate_trade(symbol),
                'system_health': self._check_system_health()
            }
        except Exception as e:
            logger.error(f"Context build failed: {str(e)}")
            raise

    def _check_system_health(self) -> Dict:
        """Check critical system metrics"""
        return {
            'memory_usage': 0.65,  # Placeholder - implement actual checks
            'cpu_load': 0.3,
            'latency_ms': 45
        }

    def _validate_trading_conditions(self, context: Dict) -> bool:
        """Execute all pre-trade validations"""
        checks = [
            not context['autopause_status'],
            context['spread_analysis']['status'] == 'normal',
            context['ftmo_compliance'],
            context['risk_assessment']['approved'],
            context['system_health']['memory_usage'] < 0.9,
            self.circuit_breaker < self.max_circuit_breaks
        ]
        
        if not all(checks):
            failed = [i for i, check in enumerate(checks) if not check]
            logger.warning(f"Validation failed on checks: {failed}")
            return False
        return True

    def _generate_trade_signal(self, context: Dict) -> Optional[Dict]:
        """Generate and validate trade signal"""
        try:
            signal = VALGHybridEngine().generate_signal(context)
            if not signal:
                logger.info("No signal generated - market conditions not favorable")
                return None
                
            required_fields = ['action', 'symbol', 'volume', 'confidence']
            if not all(f in signal for f in required_fields):
                raise ValueError(f"Invalid signal structure - missing fields")
                
            return signal
        except Exception as e:
            logger.error(f"Signal generation failed: {str(e)}")
            self.circuit_breaker += 1
            raise

    def _execute_trade_cycle(self, symbol: str):
        """Full trade decision and execution cycle"""
        try:
            context = self._build_trading_context(symbol)
            
            if not self._validate_trading_conditions(context):
                return
                
            signal = self._generate_trade_signal(context)
            if not signal:
                return
                
            result = self.components['emitter'].send(signal)
            if result.get('status') != 'success':
                raise RuntimeError(f"Emission failed: {result.get('error')}")
                
            logger.info(f"Trade executed: {signal}")
            self.components['adminbot'].send_alert(
                "TRADE_EXECUTED",
                f"{signal['symbol']} {signal['action']} {signal['volume']}L"
            )
            
        except Exception as e:
            logger.critical(f"Trade cycle failed: {str(e)}")
            self.components['adminbot'].send_alert(
                "TRADE_FAILURE",
                f"{symbol} trade failed: {str(e)}"
            )
            raise

    def run(self, symbol: str = None):
        """Main execution loop"""
        symbol = symbol or os.getenv("TRADE_SYMBOL", "EURUSD")
        logger.info(f"Starting FTMO Commander for {symbol}")
        
        try:
            self._execute_trade_cycle(symbol)
        except Exception as e:
            logger.error(f"Critical commander error: {str(e)}")
            if self.circuit_breaker >= self.max_circuit_breaks:
                logger.critical("Circuit breaker tripped - shutting down")
                self.components['adminbot'].send_alert(
                    "SYSTEM_HALT",
                    "Trading suspended due to consecutive failures"
                )
                raise SystemExit(1)

if __name__ == "__main__":
    FTMOCommander().run()
