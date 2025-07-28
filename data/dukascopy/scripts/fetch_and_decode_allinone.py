#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dukascopy Tick Data Fetcher & Decoder
Lädt BI5-Files für mehrere Tage & konvertiert zu CSV.
"""

import os
import struct
import lzma
import requests
from datetime import datetime, timedelta
import pandas as pd

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
SYMBOL = "EURUSD"
YEAR = 2025
MONTH = 7
DAYS_BACK = 3  # Anzahl der letzten Tage
SAVE_DIR = "/opt/coreflow/data/dukascopy"

def download_bi5(symbol, year, month, day, hour):
    url = f"{BASE_URL}/{symbol}/{year:04d}/{month-1:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
    local_path = f"{SAVE_DIR}/bi5_raw/{symbol}_{year:04d}{month:02d}{day:02d}_{hour:02d}.bi5"
    if os.path.exists(local_path):
        return local_path
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
        return local_path
    except Exception as e:
        print(f"❌ Fehler beim Laden {url}: {e}")
        return None

def decode_bi5_to_csv(path_bi5, out_csv):
    try:
        with open(path_bi5, "rb") as f:
            raw = lzma.decompress(f.read())
        records = []
        base_time = datetime.strptime(os.path.basename(path_bi5).split("_")[1], "%Y%m%d")
        for i in range(0, len(raw), 20):
            chunk = raw[i:i+20]
            if len(chunk) < 20:
                continue
            rel_time_ms, ask, bid, volume = struct.unpack(">QIII", chunk)
            timestamp = base_time.timestamp() + (rel_time_ms / 1000.0)
            records.append([timestamp, ask / 100000.0, bid / 100000.0, volume])
        if records:
            df = pd.DataFrame(records, columns=["timestamp", "ask", "bid", "volume"])
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
            df.to_csv(out_csv, index=False)
            print(f"✅ CSV geschrieben: {out_csv} ({len(df)} Ticks)")
        else:
            print(f"⚠️ Keine gültigen Ticks: {os.path.basename(path_bi5)}")
    except Exception as e:
        print(f"❌ Fehler bei {path_bi5}: {e}")

def main():
    now = datetime.utcnow()
    for delta in range(DAYS_BACK):
        day = now - timedelta(days=delta)
        for hour in range(24):
            bi5_path = download_bi5(SYMBOL, day.year, day.month, day.day, hour)
            if bi5_path:
                out_csv = f"{SAVE_DIR}/csv_ticks/{SYMBOL}_{day.year:04d}{day.month:02d}{day.day:02d}_{hour:02d}.csv"
                if not os.path.exists(out_csv):
                    decode_bi5_to_csv(bi5_path, out_csv)

if __name__ == "__main__":
    main()
