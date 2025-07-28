# ai_feature_engineering.py

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def extract_features_from_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    try:
        df = ohlcv.copy()
        df = df[df['close'].between(0.1, 10000)]

        returns = df['close'].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
        std_dev = returns.std()
        z_limit = 4 * std_dev if std_dev > 0 else 1
        df['return'] = returns.clip(lower=-z_limit, upper=z_limit)

        df['volatility'] = df['return'].rolling(10).std().clip(0, 1).fillna(0)
        df['spread'] = (df['high'] - df['low']).clip(0, 100).fillna(0)

        return df[['open', 'high', 'low', 'close', 'volume', 'return', 'volatility', 'spread']].dropna()
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return pd.DataFrame()
