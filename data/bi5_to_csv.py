#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bi5_to_csv.py – Dukascopy Tick-Konverter für CoreFlow
✅ Öffnet .bi5-Dateien direkt (kein gzip)
✅ Struktur: [Q rel_time_ms, I ask, I bid, I volume]
✅ Filtert ungültige Ticks (z. B. zu großer Zeitversatz)
✅ Exportiert Tickdaten als CSV (LSTM-kompatibel)
"""

import os
import struct
import csv
from datetime import datetime, timedelta

# === Verzeichnisse ===
bi5_dir = "./dukascopy_ticks"
csv_dir = "./csv_ticks"
os.makedirs(csv_dir, exist_ok=True)

def parse_bi5_file(filepath, date: datetime, hour: int):
    try:
        with open(filepath, "rb") as f:
            data = f.read()

        ticks = []
        for i in range(0, len(data), 20):
            chunk = data[i:i+20]
            if len(chunk) != 20:
                continue
            try:
                # Format: 8 Byte Zeit (Q), 3 × 4 Byte (I)
                rel_time_ms, ask, bid, volume = struct.unpack(">QIII", chunk)

                # Ungültige Ticks (z. B. rel_time_ms > 3600000) überspringen
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
        return ticks
    except Exception as e:
        print(f"❌ Fehler beim Parsen: {filepath} -> {e}")
        return []

# === Alle .bi5-Dateien im Ordner durchgehen ===
for filename in os.listdir(bi5_dir):
    if not filename.endswith(".bi5"):
        continue

    parts = filename.replace(".bi5", "").split("_")
    if len(parts) != 3:
        continue

    symbol, datestr, hourstr = parts
    try:
        date = datetime.strptime(datestr, "%Y%m%d")
        hour = int(hourstr)
    except ValueError:
        print(f"⚠️ Ungültiger Dateiname: {filename}")
        continue

    path = os.path.join(bi5_dir, filename)
    ticks = parse_bi5_file(path, date, hour)

    if ticks:
        csv_path = os.path.join(csv_dir, f"{symbol}_{datestr}_{hourstr}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "bid", "ask", "volume"])
            writer.writerows(ticks)
        print(f"✅ CSV geschrieben: {csv_path}")
    else:
        print(f"⚠️ Keine gültigen Ticks in: {filename}")

print("\n✅ Alle BI5-Dateien konvertiert.")
