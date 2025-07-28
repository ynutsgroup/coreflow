import pandas as pd
import struct
import os
from datetime import datetime, timedelta

def parse_bi5(filepath):
    with open(filepath, 'rb') as f:
        data = f.read()
    
    ticks = []
    epoch = datetime(1970, 1, 1)
    
    for i in range(0, len(data), 20):
        try:
            # Big-endian Format: 1 unsigned long long (8B) + 3 doubles (8B each)
            tick = struct.unpack('>Q3d', data[i:i+20])
            
            # Convert timestamp (milliseconds to datetime)
            timestamp = epoch + timedelta(milliseconds=tick[0])
            
            ticks.append({
                'Time': timestamp,
                'Ask': tick[1],
                'Bid': tick[2],
                'Volume': int(tick[3])
            })
        except Exception as e:
            print(f"Skipping corrupt tick at position {i}: {str(e)}")
            continue
    
    return pd.DataFrame(ticks)

if __name__ == "__main__":
    # Beispielaufruf - passen Sie den Pfad an
    sample_file = "2025/06/22/00h_ticks.bi5"
    
    if os.path.exists(sample_file):
        print(f"Verarbeite Datei: {sample_file}")
        df = parse_bi5(sample_file)
        
        if not df.empty:
            output_file = "eurusd_ticks.csv"
            df.to_csv(output_file, index=False)
            print(f"✅ Erfolg! {len(df)} Ticks in {output_file} gespeichert.")
        else:
            print("⚠️ Keine gültigen Ticks gefunden!")
    else:
        print(f"⚠️ Datei nicht gefunden: {sample_file}")
        print("Aktuelles Verzeichnis:", os.getcwd())
        print("Verfügbare Dateien:", os.listdir('.'))
