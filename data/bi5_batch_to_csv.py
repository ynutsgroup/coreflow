#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bi5_batch_to_csv.py – Konvertiert alle gültigen BI5-Dateien in CSV
"""

import os
import lzma
import struct
from datetime import datetime
import pandas as pd

bi5_dir = "./dukascopy_ticks"
csv_dir = "./csv_ticks"
os.makedirs(csv_dir, exist_ok=True)

def decode_bi5_to_df(path: str) -> pd.DataFrame:
    try:
        with lzma.open(path) as f:
            data = f.read()
        records = []
        for i in range(0, len(data), 20):
            chunk = data[i:i+20]
            if len(chunk) != 20:
                continue
            try:
                rel_time_ms, ask, bid, volume = struct.unpack(">QIII", chunk)
                timestamp = rel_time_ms / 1000.0
                if 946684800 < timestamp < 4102444800:  # 2000-01-01 bis 2100-01-01
                    records.append([timestamp, ask / 1e5, bid / 1e5, volume])
            except struct.error:
                continue
        return pd.DataFrame(records, columns=["timestamp", "ask", "bid", "volume"])
    except Exception as e:
        print(f"❌ Fehler bei {path}: {e}")
        return pd.DataFrame()

for filename in sorted(os.listdir(bi5_dir)):
    if filename.endswith(".bi5"):
        full_path = os.path.join(bi5_dir, filename)
        symbol, ymd, hour = filename.replace(".bi5", "").split("_")
        output_path = os.path.join(csv_dir, f"{symbol}_{ymd}_{hour}.csv")
        if os.path.exists(output_path):
            print(f"⏩ Bereits konvertiert: {filename}")
            continue
        df = decode_bi5_to_df(full_path)
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")
            df = df.dropna(subset=["datetime"])
            df.to_csv(output_path, index=False)
            print(f"✅ CSV geschrieben: {output_path} ({len(df)} Ticks)")
        else:
            print(f"⚠️ Keine gültigen Ticks: {filename}")
