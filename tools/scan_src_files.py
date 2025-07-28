#!/usr/bin/env python3
import os

src_path = "/opt/coreflow/src"

if not os.path.exists(src_path):
    print("❌ src-Verzeichnis nicht gefunden.")
    exit(1)

files = []
for f in os.listdir(src_path):
    full_path = os.path.join(src_path, f)
    if os.path.isfile(full_path):
        files.append(f)

print("📦 Gefundene src-Dateien:")
for f in sorted(files):
    print(f"  - {f}")
