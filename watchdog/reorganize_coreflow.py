# reorganize_coreflow.py – CoreFlow Struktur-Neuordnungsskript
# Autor: GPT CoreFlow Assist, Stand: 25. Mai 2025

import os
import shutil
from pathlib import Path

base = Path("/opt/coreflow")

structure = {
    "core/strategy": ["entry_decision_engine.py", "trendfilter.py", "orderblock.py", "killzone.py"],
    "core/ai": ["ftmo_ai_trader.py"],
    "utils": ["signal_emitter.py", "telegram_notifier.py", "lot_calculator.py", "mqtt_subscriber.py", "git_sync.sh"],
    "logs": [f.name for f in (base / "logs").glob("*.log")] if (base / "logs").exists() else [],
    ".": ["coreflow_main.py", "coreflow_watchdog.py"]
}

# Verzeichnisse anlegen
for folder in structure:
    target_path = base / folder
    target_path.mkdir(parents=True, exist_ok=True)

# Dateien verschieben, falls sie existieren
for folder, files in structure.items():
    for filename in files:
        source = base / filename
        if not source.exists():
            source = base / "utils" / filename
        target = base / folder / filename
        if source.exists() and not target.exists():
            print(f"📁 {filename} → {folder}")
            shutil.move(str(source), str(target))
        else:
            print(f"⚠️ {filename} nicht gefunden oder bereits verschoben.")

print("\n✅ CoreFlow Struktur wurde aktualisiert.")
