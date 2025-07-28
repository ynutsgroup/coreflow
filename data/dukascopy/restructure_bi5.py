import pandas as pd
import pybi5  # Stelle sicher, dass du pybi5 installiert hast: `pip install pybi5`
from pathlib import Path

# 1. Lade eine .bi5-Datei (Beispiel: erste heruntergeladene Stunde)
bi5_file = Path("2025/06/22/00h_ticks.bi5")  # Pfad anpassen!
with open(bi5_file, "rb") as f:
    ticks = pybi5.parse(f.read())  # Dekodiere die binären Daten

# 2. Konvertiere in DataFrame
df = pd.DataFrame(ticks, columns=["Time", "Ask", "Bid", "Volume"])

# 3. Speichere als CSV
df.to_csv("eurusd_ticks.csv", index=False)
print("✅ BI5 erfolgreich zu CSV konvertiert!")
