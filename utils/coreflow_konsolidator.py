#!/usr/bin/env python3
import hashlib, os, json
from pathlib import Path
from datetime import datetime

SCAN_DIRS = [
    "/opt/coreflow",
    "/opt/coreflow_ki",
    "/opt/coreflow_memory",
    "/opt/redis"
]

EXT_WHITELIST = {".py", ".sh", ".json", ".yaml", ".yml", ".env", ".txt", ".md"}
HASH_INDEX = {}
DUPLICATES = []

def get_sha256(filepath):
    try:
        with open(filepath, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return None

print(f"🔍 Starte Duplikatscan in: {', '.join(SCAN_DIRS)}")

for basedir in SCAN_DIRS:
    for root, _, files in os.walk(basedir):
        for file in files:
            p = Path(root) / file
            if p.suffix.lower() in EXT_WHITELIST:
                h = get_sha256(p)
                if not h: continue
                if h not in HASH_INDEX:
                    HASH_INDEX[h] = []
                HASH_INDEX[h].append(str(p))

for h, paths in HASH_INDEX.items():
    if len(paths) > 1:
        DUPLICATES.append({"sha256": h, "files": paths})

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path_json = f"/opt/coreflow/backup/duplicate_report_{timestamp}.json"
report_path_txt = f"/opt/coreflow/backup/duplicate_report_{timestamp}.txt"

os.makedirs("/opt/coreflow/backup/", exist_ok=True)

with open(report_path_json, "w", encoding="utf-8") as f:
    json.dump(DUPLICATES, f, indent=2)

with open(report_path_txt, "w", encoding="utf-8") as f:
    for group in DUPLICATES:
        f.write(f"\n🔁 DUPLICATE HASH: {group['sha256']}\n")
        for path in group["files"]:
            f.write(f" - {path}\n")

print(f"\n✅ Duplikatscan abgeschlossen.")
print(f"📄 Bericht gespeichert unter:\n→ {report_path_json}\n→ {report_path_txt}")
