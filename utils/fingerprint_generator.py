#!/usr/bin/env python3
# CoreFlow Fingerprint Generator – für Watchdog-Kompatibilität

import hashlib
import os
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path("/opt/coreflow")
FINGERPRINT_DIR = BASE_DIR / "fingerprints"
FINGERPRINT_DIR.mkdir(parents=True, exist_ok=True)

def generate_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def save_fingerprint(target_path: Path):
    abs_path = BASE_DIR / target_path
    if not abs_path.exists():
        print(f"❌ Datei nicht gefunden: {abs_path}")
        return

    rel_path_str = str(target_path).replace("/", "_")
    fingerprint_path = FINGERPRINT_DIR / f"{rel_path_str}.sha256"

    sha256 = generate_sha256(abs_path)
    with fingerprint_path.open("w") as f:
        f.write(f"# File: {target_path}\n")
        f.write(f"# Generated: {datetime.now(timezone.utc).isoformat()} UTC\n")
        f.write(f"SHA256={sha256}\n")

    print(f"✅ Fingerprint gespeichert: {fingerprint_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Verwendung: python3 utils/fingerprint_generator.py <relativer Pfad zur Datei>")
        print("Beispiel: python3 utils/fingerprint_generator.py core/ki/valg_engine_cpu.py")
        exit(1)

    rel_file = Path(sys.argv[1])
    save_fingerprint(rel_file)
import os
from pathlib import Path

def resolve_path(input_path):
    """Handhabt relative/absolute Pfade"""
    if input_path.startswith('/'):
        return Path(input_path)
    return Path('/opt/coreflow') / input_path
