#!/usr/bin/env python3
# /opt/coreflow/utils/coreflow_disk_audit.py

import os
import hashlib
import time
from pathlib import Path

EXCLUDE_DIRS = {'/proc', '/sys', '/dev', '/run', '/mnt', '/media', '/snap', '/var/lib/docker'}
START_PATH = "/opt/coreflow"

summary = {
    "total_files": 0,
    "total_dirs": 0,
    "total_size_gb": 0.0,
    "duplicates": [],
    "largest_files": []
}

hash_map = {}
file_sizes = []

def hash_file(path, block_size=65536):
    hasher = hashlib.md5()
    try:
        with open(path, 'rb') as f:
            while chunk := f.read(block_size):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None

def scan(path: str):
    for root, dirs, files in os.walk(path):
        # Skip excluded dirs
        if any(root.startswith(ex) for ex in EXCLUDE_DIRS):
            continue

        summary["total_dirs"] += 1

        for name in files:
            fpath = os.path.join(root, name)
            try:
                fsize = os.path.getsize(fpath)
                summary["total_files"] += 1
                summary["total_size_gb"] += fsize / (1024**3)
                file_sizes.append((fsize, fpath))

                # Duplicate-Check
                h = hash_file(fpath)
                if h:
                    if h in hash_map:
                        summary["duplicates"].append((fpath, hash_map[h]))
                    else:
                        hash_map[h] = fpath
            except Exception:
                continue

start = time.time()
scan(START_PATH)
elapsed = time.time() - start

# Top 10 große Dateien
file_sizes.sort(reverse=True)
summary["largest_files"] = file_sizes[:10]

# Output
print("\n📦 CoreFlow Disk Scan Report")
print("-" * 40)
print(f"Pfad gescannt: {START_PATH}")
print(f"Gesamtdateien: {summary['total_files']}")
print(f"Verzeichnisse:  {summary['total_dirs']}")
print(f"Größe:          {summary['total_size_gb']:.2f} GB")
print(f"Duplikate:      {len(summary['duplicates'])}")
print(f"Dauer:          {elapsed:.2f} Sek\n")

print("🔁 Doppelte Dateien:")
for dup1, dup2 in summary["duplicates"][:5]:
    print(f" - {dup1}\n   ≅ {dup2}")

print("\n📂 Top 5 größte Dateien:")
for size, path in summary["largest_files"][:5]:
    print(f" - {path} ({size / (1024**2):.2f} MB)")
