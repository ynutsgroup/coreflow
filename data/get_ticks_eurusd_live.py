#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_ticks_eurusd_live.py – Institutioneller 3-Tage-Batch für aktiven EURUSD-Handel
"""

from datetime import datetime, timedelta
from dukascopy_downloader import download_tick_hour

symbol = "EURUSD"
start_date = datetime(2025, 7, 23)
end_date   = datetime(2025, 7, 25)

# Stunden für London/NY (aktive Marktzeit)
hours = list(range(8, 18))  # 08:00–17:00 Uhr

current_date = start_date
while current_date <= end_date:
    for hour in hours:
        dt = current_date.replace(hour=hour)
        download_tick_hour(symbol, dt, save_dir="./dukascopy_ticks")
    current_date += timedelta(days=1)

print("\n✅ EURUSD aktiver 3-Tage-Download abgeschlossen.")
