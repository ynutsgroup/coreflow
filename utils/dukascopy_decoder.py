#!/usr/bin/env python3
import os
import gzip
import struct
import pandas as pd
from datetime import datetime, timedelta
import argparse

def decode_bi5(filepath):
    with open(filepath, "rb") as f:
        raw = f.read()
    try:
        raw = gzip.decompress(raw)
    except OSError:
        pass  # Datei war eventuell schon entpackt

    records = []
    for i in range(0, len(raw), 20):
        chunk = raw[i:i+20]
        if len(chunk) < 20:
            continue
        timestamp, ask, bid, volume, flags = struct.unpack(">IffHH", chunk)
        records.append((timestamp, ask, bid, volume, flags))
    return records

def decode_day(symbol, date, export=False):
    base_path = f"/opt/coreflow/data/dukascopy/{symbol}/{date.replace('-', '/')}"
    all_ticks = []
    for h in range(24):
        filename = f"{h:02d}h_ticks.bi5"
        fullpath = os.path.join(base_path, filename)
        if os.path.exists(fullpath):
            hourly_ticks = decode_bi5(fullpath)
            all_ticks.extend(hourly_ticks)

    df = pd.DataFrame(all_ticks, columns=["rel_time", "ask", "bid", "volume", "flags"])
    if df.empty:
        return df

    base_time = datetime.strptime(date, "%Y-%m-%d")
    df["timestamp"] = df["rel_time"].apply(lambda x: base_time + timedelta(milliseconds=x))
    df = df.drop(columns=["rel_time"])
    df = df[["timestamp", "ask", "bid", "volume", "flags"]]

    if export:
        outpath = f"/opt/coreflow/data/decoded/{symbol}_{date}.csv"
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        df.to_csv(outpath, index=False)
        print(f"✅ Export: {outpath}")

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode Dukascopy BI5 ticks")
    parser.add_argument("symbol", help="e.g. EURUSD")
    parser.add_argument("date", help="Format: YYYY-MM-DD")
    parser.add_argument("--export", action="store_true", help="Export CSV")
    args = parser.parse_args()

    df = decode_day(args.symbol, args.date, export=args.export)
    print(df.head() if not df.empty else "⚠️ No data found.")
