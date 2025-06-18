import numpy as np
import pandas as pd
from typing import Tuple, Optional

class FTMO_VALGEngine:
    def __init__(self, 
                 account_size: float = 10000,
                 max_risk_per_trade: float = 0.01,  # 1% Risiko pro Trade (FTMO Standard)
                 volatility_window: int = 34,       # Fibonacci-basiertes Fenster
                 min_volume_spike: float = 2.0):    # 2x durchschn. Volumen
        """
        FTMO-konformer VALG Engine mit striktem Risikomanagement
        """
        self.max_risk = min(max_risk_per_trade, 0.02)  # Hardcap bei 2%
        self.min_stop_loss_pips = 10  # Mindest-Stop-Loss (FTMO Regel)
        self.max_daily_trades = 5     # Max Trades/Tag (Anti-Overtrading)
        self.account_size = account_size
        self.volatility_window = volatility_window
        self.min_volume_spike = min_volume_spike
        self._trade_count = 0
        self._last_signal = {
            'direction': 0,
            'size': 0,
            'stop_loss': 0,
            'take_profit': 0
        }

    def _calculate_volatility(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        tr = pd.concat([
            (high - low).abs(),
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(span=self.volatility_window).mean()

    def _get_volume_anomalies(self, volume: pd.Series) -> pd.Series:
        median_vol = volume.rolling(50).median()
        return volume > (median_vol * self.min_volume_spike)

    def _get_trade_size(self, volatility: float, price: float) -> float:
        risk_amount = self.account_size * self.max_risk
        stop_loss_distance = max(volatility * 1.5, self.min_stop_loss_pips * 0.0001)
        return round(risk_amount / (price * stop_loss_distance), 2)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['volatility'] = self._calculate_volatility(df['high'], df['low'], df['close'])
        df['volume_anomaly'] = self._get_volume_anomalies(df['volume'])

        long_cond = (
            df['volume_anomaly'] & 
            (df['close'] < df['open']) & 
            (df['volatility'] < df['volatility'].quantile(0.3))
        )
        short_cond = (
            df['volume_anomaly'] & 
            (df['close'] > df['open']) & 
            (df['volatility'] < df['volatility'].quantile(0.3))
        )

        df['signal'] = 0
        df['size'] = 0.0
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan

        for i in range(1, len(df)):
            if self._trade_count >= self.max_daily_trades:
                break

            if long_cond.iloc[i]:
                sl = df['low'].iloc[i] - (df['volatility'].iloc[i] * 1.5)
                tp = df['close'].iloc[i] + (3 * df['volatility'].iloc[i])
                size = self._get_trade_size(df['volatility'].iloc[i], df['close'].iloc[i])

                df.at[df.index[i], 'signal'] = 1
                df.at[df.index[i], 'size'] = size
                df.at[df.index[i], 'stop_loss'] = sl
                df.at[df.index[i], 'take_profit'] = tp
                self._trade_count += 1

            elif short_cond.iloc[i]:
                sl = df['high'].iloc[i] + (df['volatility'].iloc[i] * 1.5)
                tp = df['close'].iloc[i] - (3 * df['volatility'].iloc[i])
                size = self._get_trade_size(df['volatility'].iloc[i], df['close'].iloc[i])

                df.at[df.index[i], 'signal'] = -1
                df.at[df.index[i], 'size'] = size
                df.at[df.index[i], 'stop_loss'] = sl
                df.at[df.index[i], 'take_profit'] = tp
                self._trade_count += 1

        last_idx = df[df['signal'] != 0].last_valid_index()
        if last_idx:
            self._last_signal = {
                'direction': df.at[last_idx, 'signal'],
                'size': df.at[last_idx, 'size'],
                'stop_loss': df.at[last_idx, 'stop_loss'],
                'take_profit': df.at[last_idx, 'take_profit'],
                'timestamp': df.index[-1],
                'symbol': df.columns.name if df.columns.name else "UNKNOWN",
                'entry': df['close'].iloc[-1]
            }

        return df

    def reset_daily_count(self):
        self._trade_count = 0

    @property
    def last_signal(self) -> dict:
        return self._last_signal

    def get_ftmo_rules(self) -> dict:
        return {
            'max_risk_per_trade': self.max_risk,
            'min_stop_loss_pips': self.min_stop_loss_pips,
            'max_daily_trades': self.max_daily_trades,
            'risk_reward_ratio': '1:3',
            'allowed_strategies': ['VALG_Liquidity'],
            'prohibited': ['Martingale', 'Grid', 'Hedging']
        }

    async def execute_trade(self, signal: dict, risk_percent: float, account_size: float) -> dict:
        """
        Dummy Trade-Ausführung – simuliert PnL und gibt Resultat zurück.
        """
        direction = "BUY" if signal['direction'] == 1 else "SELL"
        volume = signal['size']
        price = signal.get("entry", 1.0)
        sl = signal.get("stop_loss", 0)
        tp = signal.get("take_profit", 0)

        simulated_pnl = round(np.random.uniform(-1, 3), 2) * risk_percent * account_size

        result = {
            "symbol": signal.get("symbol", "UNKNOWN"),
            "direction": direction,
            "volume": volume,
            "entry": price,
            "stop_loss": sl,
            "take_profit": tp,
            "pnl": simulated_pnl,
            "risk": risk_percent,
            "account_size": account_size,
            "timestamp": signal.get("timestamp")
        }

        print(f"✅ Simulierter Trade ausgeführt: {result}")
        return result
