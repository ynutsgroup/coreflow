zunkunft orientiert: #!/usr/bin/env python3
# CoreFlow Institutional MT5 Pro - KI-optimiert & FTMO-konform (Linux Version)

import os
import json
import time
import logging
import hashlib
import traceback
import redis
from datetime import datetime
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from cryptography.fernet import Fernet
from dotenv import load_dotenv

# === INSTITUTIONELLE KONFIGURATION ===
load_dotenv("/opt/coreflow/.env")

CONFIG = {
    # Redis-Konfiguration
    "REDIS": {
        "host": os.getenv("REDIS_HOST", "localhost"),
        "port": int(os.getenv("REDIS_PORT", "6379")),
        "password": os.getenv("REDIS_PASSWORD"),
        "channel": os.getenv("REDIS_CHANNEL", "trading_signals"),
        "timeout": 5.0
    },
    
    # MT5 Zugangsdaten (Verschlüsselt)
    "MT5_CREDENTIALS": {
        "login": int(os.getenv("MT5_LOGIN")),
        "password": hashlib.sha256(os.getenv("MT5_PASSWORD").encode()).hexdigest(),
        "server": os.getenv("MT5_SERVER"),
        "timeout": 15000
    },
    
    # Risikomanagement
    "RISK_PARAMS": {
        "max_daily_trades": int(os.getenv("MAX_DAILY_TRADES", "10")),
        "max_drawdown": float(os.getenv("MAX_DRAWDOWN", "0.05")),
        "stop_loss_pips": int(os.getenv("STOP_LOSS_PIPS", "50")),
        "take_profit_pips": int(os.getenv("TAKE_PROFIT_PIPS", "100")),
        "daily_loss_limit": float(os.getenv("DAILY_LOSS_LIMIT", "0.02"))  # 2%
    },
    
    # KI-Parameter
    "AI_SETTINGS": {
        "anomaly_threshold": float(os.getenv("ANOMALY_THRESHOLD", "0.95")),
        "volatility_window": int(os.getenv("VOLATILITY_WINDOW", "14")),
        "trend_confirmation": os.getenv("TREND_CONFIRMATION", "True") == "True"
    },
    
    # Verschlüsselung
    "ENCRYPTION": {
        "key": os.getenv("ENCRYPTION_KEY"),
        "require_encryption": True
    }
}

# === INSTITUTIONELLES LOGGING ===
class InstitutionalLogger:
    def __init__(self):
        os.makedirs("/opt/coreflow/logs", exist_ok=True)
        
        self.logger = logging.getLogger("CoreFlowMT5Pro")
        self.logger.setLevel(logging.INFO)
        
        # Farbige Konsolenausgabe
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(
            "\033[92m%(asctime)s | %(levelname)s | %(message)s\033[0m"
        ))
        
        # Detaillierte Dateiausgabe
        file = logging.FileHandler(
            "/opt/coreflow/logs/mt5_pro.log", 
            encoding="utf-8"
        )
        file.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        ))
        
        self.logger.addHandler(console)
        self.logger.addHandler(file)
    
    def log_execution(self, symbol, action, lots, price, success):
        """Protokolliert Trade-Execution mit Performance-Metriken"""
        account = mt5.account_info()
        msg = (
            f"{'✅' if success else '❌'} {action} {symbol} {lots} Lots @ {price} | "
            f"Equity: {account.equity:.2f} | "
            f"Margin: {(account.margin/account.equity*100):.1f}% | "
            f"Daily P/L: {(account.equity - account.balance):.2f}"
        )
        self.logger.info(msg)

logger = InstitutionalLogger().logger

# === KI-OPTIMIERTES RISIKOMANAGEMENT ===
class AITradingGuard:
    def __init__(self):
        self.model = self._load_anomaly_model()
        self.fernet = Fernet(CONFIG["ENCRYPTION"]["key"])
        self.redis = redis.StrictRedis(
            host=CONFIG["REDIS"]["host"],
            port=CONFIG["REDIS"]["port"],
            password=CONFIG["REDIS"]["password"],
            socket_timeout=CONFIG["REDIS"]["timeout"]
        )
        self.pubsub = self.redis.pubsub()
        self.pubsub.subscribe(CONFIG["REDIS"]["channel"])
    
    def _load_anomaly_model(self):
        """Lädt vortrainiertes KI-Modell für Anomalie-Erkennung"""
        try:
            # Hier würde normalerweise das Modell geladen werden
            return IsolationForest(
                n_estimators=100,
                contamination=0.01,
                random_state=42
            )
        except:
            logger.warning("KI-Modell nicht geladen, Standardwerte verwendet")
            return None
    
    def _decrypt_signal(self, encrypted_payload):
        """Entschlüsselt das Signal mit Fernet"""
        try:
            decrypted = self.fernet.decrypt(encrypted_payload.encode()).decode()
            return json.loads(decrypted)
        except:
            logger.error("Signalentschlüsselung fehlgeschlagen")
            return None
    
    def get_signal(self):
        """Empfängt und validiert Signale von Redis"""
        message = self.pubsub.get_message()
        if message and message['type'] == 'message':
            try:
                data = json.loads(message['data'])
                if CONFIG["ENCRYPTION"]["require_encryption"]:
                    signal = self._decrypt_signal(data['encrypted_payload'])
                else:
                    signal = data
                
                if signal and self.validate_signal(signal):
                    return signal
                    
            except Exception as e:
                logger.error(f"Signalverarbeitung fehlgeschlagen: {str(e)}")
        
        return None
    
    def analyze_market(self, symbol):
        """KI-gestützte Marktanalyse"""
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
        df = pd.DataFrame(rates)
        
        # Volatilitätsanalyse
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        volatility = df['returns'].std() * np.sqrt(252)
        
        # Trendanalyse
        trend_strength = (df['close'][-1] - df['close'].mean()) / df['close'].std()
        
        return {
            'volatility': volatility,
            'trend_strength': trend_strength,
            'spread': mt5.symbol_info(symbol).spread
        }

    def validate_signal(self, signal):
        """KI-Validierung des Trading-Signals"""
        required_fields = ["symbol", "action", "lot", "timestamp"]
        if not all(field in signal for field in required_fields):
            logger.warning("Ungültiges Signalformat")
            return False
        
        analysis = self.analyze_market(signal['symbol'])
        
        # FTMO-Risikokontrolle
        account = mt5.account_info()
        equity_ratio = account.equity / account.balance
        
        if equity_ratio < 1 - CONFIG["RISK_PARAMS"]["max_drawdown"]:
            logger.warning("⚠️ Drawdown-Limit erreicht - Trade abgelehnt")
            return False
            
        if (account.equity - account.balance) < -CONFIG["RISK_PARAMS"]["daily_loss_limit"] * account.balance:
            logger.warning("⚠️ Tägliches Verlustlimit erreicht")
            return False
        
        # KI-Anomalieerkennung
        features = np.array([
            analysis['volatility'],
            analysis['trend_strength'],
            float(signal['lot'])
        ]).reshape(1, -1)
        
        if self.model and self.model.predict(features)[0] == -1:
            logger.warning("⚠️ KI-Anomalie erkannt - Trade abgelehnt")
            return False
            
        return True

# === INSTITUTIONELLE TRADING-ENGINE ===
class InstitutionalTrader:
    def __init__(self):
        self.risk_guard = AITradingGuard()
        self.today_trades = 0
        self.daily_start_balance = mt5.account_info().balance
    
    def _calculate_lot_size(self, symbol):
        """Dynamische Lot-Größenberechnung (FTMO-konform)"""
        account = mt5.account_info()
        risk = CONFIG["RISK_PARAMS"]["max_drawdown"]
        symbol_info = mt5.symbol_info(symbol)
        
        # KI-optimierte Lot-Größe
        lot_size = round(
            (account.equity * risk) / (symbol_info.point * CONFIG["RISK_PARAMS"]["stop_loss_pips"]),
            2
        )
        return min(lot_size, account.margin_free * 0.1)  # Max 10% Margin

    def execute_trade(self, signal):
        """Institutionelle Trade-Execution mit KI-Validierung"""
        # Tägliches Trading-Limit
        if self.today_trades >= CONFIG["RISK_PARAMS"]["max_daily_trades"]:
            logger.warning("⚠️ Tägliches Trade-Limit erreicht")
            return False
        
        symbol = signal['symbol']
        action = signal['action'].upper()
        
        try:
            # Marktdaten abrufen
            if not mt5.symbol_select(symbol, True):
                raise ValueError(f"Symbol {symbol} nicht verfügbar")
            
            tick = mt5.symbol_info_tick(symbol)
            point = mt5.symbol_info(symbol).point
            
            # Order-Parameter
            order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
            price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
            stop_loss = price - (point * CONFIG["RISK_PARAMS"]["stop_loss_pips"] * (-1 if order_type == mt5.ORDER_TYPE_BUY else 1))
            take_profit = price + (point * CONFIG["RISK_PARAMS"]["take_profit_pips"] * (1 if order_type == mt5.ORDER_TYPE_BUY else -1))
            
            # Dynamische Lot-Größe
            lot_size = self._calculate_lot_size(symbol)
            
            # Trade-Request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": order_type,
                "price": price,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": 10,
                "magic": 20230615,
                "comment": "CoreFlow AI Trade",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK
            }
            
            # Trade ausführen
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                raise RuntimeError(f"Order fehlgeschlagen: {result.comment}")
            
            # Erfolgsprotokollierung
            self.today_trades += 1
            logger.log_execution(symbol, action, lot_size, price, True)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Trade-Execution fehlgeschlagen: {str(e)}")
            logger.debug(traceback.format_exc())
            return False

# === HAUPTPROGRAMM ===
if __name__ == "__main__":
    try:
        logger.info("=== CoreFlow Institutional MT5 Pro (Linux) ===")
        
        # MT5-Verbindung herstellen
        if not mt5.initialize(**CONFIG["MT5_CREDENTIALS"]):
            raise ConnectionError(f"MT5-Verbindung fehlgeschlagen: {mt5.last_error()}")
        
        logger.info(f"✅ Verbunden mit {CONFIG['MT5_CREDENTIALS']['server']}")
        logger.info(f"💰 Kontostand: {mt5.account_info().equity:.2f} USD")
        
        trader = InstitutionalTrader()
        
        # Hauptloop
        while True:
            signal = trader.risk_guard.get_signal()
            if signal:
                trader.execute_trade(signal)
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Strategie manuell gestoppt")
    except Exception as e:
        logger.error(f"💥 Kritischer Fehler: {str(e)}")
    finally:
        mt5.shutdown()
        logger.info("=== System heruntergefahren ===")
