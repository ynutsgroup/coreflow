#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COREFLOW TIMESERIES PLOTTER v3.1
✅ Dynamische Konfiguration | ✅ KI-ready | ✅ FTMO-konform
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Tuple, Union
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from pathlib import Path
import sys

# 🔐 Dynamische Konfiguration
try:
    decrypt_path = Path("/opt/coreflow/utils")
    if decrypt_path.exists():
        sys.path.insert(0, str(decrypt_path))
    else:
        raise FileNotFoundError(f"decrypt_env.py nicht gefunden: {decrypt_path}")

    from decrypt_env import load_env, find_latest_env_enc

    env_path = find_latest_env_enc()
    if not env_path:
        raise FileNotFoundError("Keine gültige .env.enc-Datei gefunden")
    env = load_env(env_path, "/opt/coreflow/infra/vault/encryption.key")

    from modules.timeseries.timeseries_interface import ts_client
except Exception as e:
    logging.critical(f"Konfigurationsfehler: {str(e)}")
    sys.exit(1)

# Logging Konfiguration
logger = logging.getLogger('coreflow.plotter')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(module)s | %(message)s"))
logger.addHandler(handler)

class InstitutionalPlotter:
    """
    Institutioneller Plotter für Trading-Analysen mit:
    - Dynamischer Konfiguration
    - KI-optimierten Features
    - FTMO-Compliance
    """

    def __init__(self):
        self.style_config = {
            'dark': {
                'bg_color': '#1a1a1a',
                'text_color': '#e0e0e0',
                'grid_color': '#2a2a2a',
                'price_color': '#4e79a7',
                'indicators': {
                    'sma': '#f28e2b',
                    'ema': '#59a14f',
                    'bollinger': '#e15759',
                    'rsi': '#b07aa1'
                }
            },
            'light': {
                'bg_color': '#ffffff',
                'text_color': '#000000',
                'grid_color': '#f0f0f0',
                'price_color': '#1f77b4',
                'indicators': {
                    'sma': '#ff7f0e',
                    'ema': '#2ca02c',
                    'bollinger': '#d62728',
                    'rsi': '#9467bd'
                }
            }
        }

    def plot_series(
        self,
        symbol: str,
        timeframe: str = "1D",
        lookback: Union[int, timedelta] = 30,
        indicators: Optional[List[str]] = None,
        style: str = "dark",
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (16, 8)
    ) -> Optional[Figure]:
        """
        Erweitertes Plotting mit institutionellen Features
        """
        try:
            start_date = self._calculate_start_date(lookback)
            df = self._get_data(symbol, start_date, timeframe)

            if df.empty:
                logger.warning(f"Keine Daten für {symbol} verfügbar")
                return None

            style_params = self.style_config.get(style, self.style_config['dark'])
            fig, axes = self._create_figure(indicators, figsize, style_params)

            self._plot_price(axes[0], df, symbol, timeframe, style_params)

            if indicators:
                self._add_indicators(axes[0], df, indicators, style_params)

            if indicators and 'rsi' in indicators and len(axes) > 1:
                self._add_rsi(axes[1], df, style_params)

            self._finalize_plot(fig, axes, symbol, timeframe, style_params)

            if save_path:
                self._save_plot(fig, save_path)

            if show:
                plt.show()

            return fig

        except Exception as e:
            logger.error(f"Plot-Fehler für {symbol}: {str(e)}", exc_info=True)
            return None

    def _calculate_start_date(self, lookback: Union[int, timedelta]) -> datetime:
        if isinstance(lookback, timedelta):
            return datetime.now() - lookback
        return datetime.now() - timedelta(days=lookback)

    def _get_data(self, symbol: str, start_date: datetime, timeframe: str) -> pd.DataFrame:
        df = ts_client.get_range_df(symbol, from_ts=start_date, resample=timeframe)
        df['value'] = df['value'].apply(lambda x: float(x.decode()) if isinstance(x, bytes) else float(x))
        return df.dropna()

    def _create_figure(self, indicators: Optional[List[str]], figsize: Tuple[int, int], style: Dict) -> Tuple[Figure, List[plt.Axes]]:
        if indicators and 'rsi' in indicators:
            fig, axes = plt.subplots(
                2, 1,
                figsize=figsize,
                gridspec_kw={'height_ratios': [3, 1]},
                sharex=True
            )
            return fig, axes
        fig, ax = plt.subplots(figsize=figsize)
        return fig, [ax]

    def _plot_price(self, ax: plt.Axes, df: pd.DataFrame, symbol: str, timeframe: str, style: Dict):
        ax.plot(
            df.index, df['value'],
            label=f'{symbol.upper()} Preis',
            color=style['price_color'],
            linewidth=2
        )
        ax.set_title(
            f"{symbol.upper()} - {timeframe} Chart",
            pad=20,
            fontsize=14,
            color=style['text_color']
        )
        ax.set_ylabel('Preis', color=style['text_color'])

    def _add_indicators(self, ax: plt.Axes, df: pd.DataFrame, indicators: List[str], style: Dict):
        if 'sma' in indicators:
            sma = df['value'].rolling(20).mean()
            ax.plot(sma.index, sma, label='SMA 20', color=style['indicators']['sma'], linestyle='--')

        if 'ema' in indicators:
            ema = df['value'].ewm(span=50).mean()
            ax.plot(ema.index, ema, label='EMA 50', color=style['indicators']['ema'], linestyle='-.')

        if 'bollinger' in indicators:
            sma = df['value'].rolling(20).mean()
            std = df['value'].rolling(20).std()
            ax.plot(sma.index, sma, label='Bollinger Mittel', color=style['indicators']['bollinger'])
            ax.fill_between(
                sma.index,
                sma - 2 * std,
                sma + 2 * std,
                color=style['indicators']['bollinger'],
                alpha=0.1
            )

    def _add_rsi(self, ax: plt.Axes, df: pd.DataFrame, style: Dict):
        delta = df['value'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / loss))

        ax.plot(rsi.index, rsi, label='RSI', color=style['indicators']['rsi'])
        ax.axhline(70, color='red', linestyle='--', alpha=0.3)
        ax.axhline(30, color='green', linestyle='--', alpha=0.3)
        ax.set_ylim(0, 100)
        ax.set_ylabel('RSI', color=style['text_color'])
        ax.legend(loc='upper right')

    def _finalize_plot(self, fig: Figure, axes: List[plt.Axes], symbol: str, timeframe: str, style: Dict):
        for ax in axes:
            ax.grid(True, color=style['grid_color'], linestyle='--', alpha=0.7)
            ax.set_facecolor(style['bg_color'])
            ax.tick_params(colors=style['text_color'])
            ax.legend(loc='upper left')

            if timeframe in ['1D', '1W']:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            else:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

            ax.xaxis.set_major_locator(mdates.AutoDateLocator())

        fig.patch.set_facecolor(style['bg_color'])
        plt.xticks(rotation=45)
        plt.tight_layout()

    def _save_plot(self, fig: Figure, path: str):
        fig.savefig(
            path,
            dpi=300,
            bbox_inches='tight',
            facecolor=fig.get_facecolor()
        )
        logger.info(f"Diagramm gespeichert unter: {path}")

# Globale Instanz
ts_plotter = InstitutionalPlotter()

# Direktzugriff
def plot_series(symbol: str, **kwargs):
    return ts_plotter.plot_series(symbol, **kwargs)

# Testlauf
if __name__ == "__main__":
    ts_plotter.plot_series(
        "EURUSD",
        timeframe="1H",
        lookback=7,
        indicators=['sma', 'rsi'],
        style="dark",
        save_path="/tmp/eurusd_chart.png"
    )
