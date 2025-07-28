import os
from pathlib import Path

ARCHIVE_DIR = Path("/opt/coreflow_memory")
SESSIONS = sorted(ARCHIVE_DIR.glob("session_*"))

if len(SESSIONS) < 2:
    print("⚠️ Nicht genug Sessions für Vergleich.")
    exit(0)

last = SESSIONS[-2]
current = SESSIONS[-1]

print(f"📅 Vergleich: {last.name} ⇨ {current.name}")

FILES = ["structure_manifest.json", "coreflow_progress.log", "chat_today.md"]
for fname in FILES:
    f1 = last / fname
    f2 = current / fname
    if f1.exists() and f2.exists():
        if f1.read_bytes() != f2.read_bytes():
            print(f"⚠️ Geändert: {fname}")
        else:
            print(f"✅ Unverändert: {fname}")
    elif f2.exists():
        print(f"🆕 Neu in letzter Session: {fname}")
    elif f1.exists():
        print(f"🗑️ Entfernt in letzter Session: {fname}")
    else:
        print(f"❓ Nicht vorhanden: {fname}")
