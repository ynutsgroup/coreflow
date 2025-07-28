

#!/usr/bin/env python3

import sys
import hashlib
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: verify_fingerprint.py <file_path>")
    sys.exit(1)

file_path = Path(sys.argv[1])
fingerprint_dir = Path("/opt/coreflow/fingerprints")

# Fingerprint-Dateiname nach alter und neuer Methode
legacy_filename = "_" + file_path.as_posix().replace("/", "_") + ".sha256"
standard_filename = file_path.as_posix().replace("/opt/coreflow/", "").replace("/", "_") + ".sha256"

legacy_path = fingerprint_dir / legacy_filename
standard_path = fingerprint_dir / standard_filename

# Bestehenden Pfad wählen
if standard_path.exists():
    fingerprint_path = standard_path
elif legacy_path.exists():
    fingerprint_path = legacy_path
else:
    print("❌ Kein Fingerprint vorhanden.")
    sys.exit(2)

# SHA256 vergleichen
with open(file_path, "rb") as f:
    current_hash = hashlib.sha256(f.read()).hexdigest()

with open(fingerprint_path, "r") as f:
    saved_hash = f.readline().strip().split()[0]

if current_hash == saved_hash:
    print("✅ Fingerprint ist gültig.")
else:
    print("❌ Fingerprint stimmt NICHT überein.")
