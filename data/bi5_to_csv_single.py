#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bi5_to_csv_single.py – wandelt genau eine .bi5-Datei in CSV um
"""

import os
import struct
import csv
from datetime import datetime, timedelta

# === Konfiguration ===
filename = "EURUSD_20250725_13.bi5"
bi5_path = f"./dukascopy_ticks/{filename}"
csv_path = f"./csv_ticks/{filename.replace('.bi5', '.csv')}"

# === Zeit aus Dateiname extrahieren
parts = filename.replace(".bi5", "").split("_")
symbol, datestr, hourstr = parts
date = datetime.strptime(datestr, "%Y%m%d")
hour = int(hourstr)

# === Konvertierung starten
ticks = []
try:
    with open(bi5_path, "rb") as f:
        data = f.read()

    for i in range(0, len(data), 20):
        chunk = data[i:i+20]
        if len(chunk) != 20:
            continue
        try:
            rel_time_ms, ask, bid, volume, _ = struct.unpack(">IIIIf", chunk)
            if rel_time_ms > 3_600_000:
                continue
            timestamp = date + timedelta(hours=hour, milliseconds=rel_time_ms)
            ticks.append([
                timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                bid / 1e5,
                ask / 1e5,
                volume
            ])
        except Exception:
            continue

    # Schreiben
    if ticks:
        os.makedirs("./csv_ticks", exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "bid", "ask", "volume"])
            writer.writerows(ticks)
        print(f"✅ CSV geschrieben: {csv_path} ({len(ticks)} Ticks)")
    else:
        print(f"⚠️ Keine gültigen Ticks in Datei.")
except Exception as e:
    print(f"❌ Fehler beim Verarbeiten: {e}")
