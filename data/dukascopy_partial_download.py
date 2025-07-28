#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dukascopy_partial_download.py – gezielte Tickdaten für aktive Handelsstunden
Zeitraum: 25.07.2025 von 08:00 bis 13:00 Uhr
"""

import os
import requests
from datetime import datetime

# === Einstellungen ===
symbol = "EURUSD"
date = datetime(2025, 7, 25)
save_dir = "./dukascopy_ticks"
os.makedirs(save_dir, exist_ok=True)

def download_tick_hour(symbol: str, dt: datetime):
    y, m, d, h = dt.year, dt.month - 1, dt.day, dt.hour
    url = f"https://datafeed.dukascopy.com/datafeed/{symbol}/{y}/{str(m).zfill(2)}/{str(d).zfill(2)}/{str(h).zfill(2)}h_ticks.bi5"
    filename = f"{symbol}_{dt.strftime('%Y%m%d')}_{str(h).zfill(2)}.bi5"
    save_path = os.path.join(save_dir, filename)

    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and len(r.content) > 1000:
            with open(save_path, "wb") as f:
                f.write(r.content)
            print(f"✅ {filename} gespeichert ({len(r.content)//1024} KB)")
        else:
            print(f"⚠️  {filename} enthält keine brauchbaren Ticks ({len(r.content)} Bytes)")
    except Exception as e:
        print(f"❌ Fehler bei {filename}: {e}")

# === Stunden 08 bis 13 herunterladen ===
for hour in range(8, 14):
    dt = date.replace(hour=hour)
    download_tick_hour(symbol, dt)

print("\n✅ Teil-Download abgeschlossen.")
