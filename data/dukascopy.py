#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dukascopy.py – Tickdaten Downloader (EURUSD)
✅ Lädt Tickdaten (bi5) für mehrere Tage und Stunden
✅ Speichert unter ./dukascopy_ticks/
"""

import os
import requests
from datetime import datetime, timedelta

# === Einstellungen ===
symbol = "EURUSD"
start_date = datetime(2025, 7, 20)
end_date   = datetime(2025, 7, 25)
save_dir = "./dukascopy_ticks"
os.makedirs(save_dir, exist_ok=True)

# === Download-Funktion für 1 Stunde ===
def download_tick_hour(symbol: str, date: datetime):
    y, m, d, h = date.year, date.month - 1, date.day, date.hour
    url = f"https://datafeed.dukascopy.com/datafeed/{symbol}/{y}/{str(m).zfill(2)}/{str(d).zfill(2)}/{str(h).zfill(2)}h_ticks.bi5"
    filename = f"{symbol}_{date.strftime('%Y%m%d')}_{str(h).zfill(2)}.bi5"
    save_path = os.path.join(save_dir, filename)

    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and len(r.content) > 1000:
            with open(save_path, "wb") as f:
                f.write(r.content)
            print(f"✅ {url} -> {filename} gespeichert ({len(r.content)//1024} KB)")
        else:
            print(f"⚠️  {url} enthält keine brauchbaren Ticks ({len(r.content)} Bytes)")
    except Exception as e:
        print(f"❌ Fehler bei {url}: {e}")

# === Alle Stunden in Zeitraum durchgehen ===
current = start_date
while current <= end_date:
    for hour in range(24):
        dt = current.replace(hour=hour)
        download_tick_hour(symbol, dt)
    current += timedelta(days=1)

print("\n✅ Alle Stunden für Zeitraum heruntergeladen.")
