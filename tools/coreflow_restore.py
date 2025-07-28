import argparse
import shutil
from pathlib import Path
import logging
import hashlib

# === Konfiguration ===
ARCHIVE_ROOT = Path("/opt/coreflow_memory")
PROJECT_DIR = Path("/opt/coreflow")
LOG_FILE = "/opt/coreflow/logs/restore.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def sha256_of_file(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(4096):
            h.update(chunk)
    return h.hexdigest()

def verify_hash(source: Path, target: Path) -> bool:
    if not source.exists():
        return False
    source_hash = sha256_of_file(source)
    if target.exists():
        target_hash = sha256_of_file(target)
        return source_hash == target_hash
    return True  # wenn Ziel noch nicht existiert, akzeptieren

def list_sessions():
    print("📂 Verfügbare Sessions:")
    for path in sorted(ARCHIVE_ROOT.glob("session_*")):
        print(" -", path.name)

def restore_file(session: str, filename: str, force: bool = False):
    src = ARCHIVE_ROOT / f"session_{session}" / filename
    dest = PROJECT_DIR / filename
    if not src.exists():
        print(f"❌ Nicht gefunden: {src}")
        return
    if not force and not verify_hash(src, dest):
        print(f"⚠️ HASH-Mismatch: {filename} – Wiederherstellung abgebrochen.")
        logging.warning(f"HASH mismatch for {filename}. Restore skipped.")
        return
    shutil.copy2(src, dest)
    print(f"✅ Wiederhergestellt: {dest}")
    logging.info(f"Datei wiederhergestellt: {filename} aus Session {session}")

def restore_full(session: str, force: bool = False):
    session_dir = ARCHIVE_ROOT / f"session_{session}"
    if not session_dir.exists():
        print("❌ Session nicht gefunden:", session)
        return
    restored = 0
    for file in session_dir.iterdir():
        if file.name.endswith(".zip"):
            continue  # ZIP nicht automatisch entpacken
        dest = PROJECT_DIR / file.name
        if not force and not verify_hash(file, dest):
            print(f"⚠️ HASH-Mismatch: {file.name} – übersprungen.")
            logging.warning(f"HASH mismatch for {file.name}")
            continue
        shutil.copy2(file, dest)
        print(f"✅ {file.name}")
        restored += 1
    logging.info(f"{restored} Dateien wiederhergestellt aus Session {session}")

# === CLI Interface ===
parser = argparse.ArgumentParser(description="CoreFlow Restore Tool (mit Hash-Check)")
parser.add_argument("--list", action="store_true", help="Alle Sessions anzeigen")
parser.add_argument("--session", type=str, help="Session-Zeitstempel z. B. 2025-07-02_16-33")
parser.add_argument("--file", type=str, help="Nur eine Datei wiederherstellen")
parser.add_argument("--full", action="store_true", help="Komplette Session wiederherstellen")
parser.add_argument("--force", action="store_true", help="Restore trotz HASH-Unterschied")

args = parser.parse_args()

if args.list:
    list_sessions()
elif args.session and args.file:
    restore_file(args.session, args.file, force=args.force)
elif args.session and args.full:
    restore_full(args.session, force=args.force)
else:
    parser.print_help()
