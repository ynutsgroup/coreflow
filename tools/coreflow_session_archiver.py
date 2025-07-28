import os
import shutil
import zipfile
import logging
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
from tools.telegram_alert import send_telegram_message

# === Logging Setup ===
LOG_PATH = "/opt/coreflow/logs/backup.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# === Konfiguration ===
class BackupConfig:
    BASE_DIR = "/opt/coreflow_memory"
    PROJECT_DIR = "/opt/coreflow"
    ZIP_NAME = "autotrader_snapshot.zip"
    RETENTION_DAYS = 30
    FILES_TO_SAVE = [
        "structure_manifest.json",
        "coreflow_progress.log",
        "chat_today.md"
    ]
    IGNORE_DIRS = {".git", "__pycache__", "venv", "tmp"}

config = BackupConfig()

# === Funktionen ===

def get_file_hash(filepath: str) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(4096):
            sha256.update(chunk)
    return sha256.hexdigest()

def backup_files(target_dir: Path) -> int:
    saved = 0
    for filename in config.FILES_TO_SAVE:
        src = Path(config.PROJECT_DIR) / filename
        dest = target_dir / filename
        try:
            if src.exists():
                shutil.copy2(src, dest)
                logger.info(f"✅ Saved: {filename} (SHA256: {get_file_hash(dest)})")
                saved += 1
            else:
                logger.warning(f"⚠️ File not found: {filename}")
        except Exception as e:
            logger.error(f"❌ Failed to copy {filename}: {e}")
    return saved

def zip_project(target_zip: Path) -> bool:
    try:
        with zipfile.ZipFile(target_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(config.PROJECT_DIR):
                dirs[:] = [d for d in dirs if d not in config.IGNORE_DIRS]
                for file in files:
                    file_path = Path(root) / file
                    rel_path = os.path.relpath(file_path, config.PROJECT_DIR)
                    zipf.write(file_path, arcname=rel_path)
        logger.info(f"📦 Project zipped: {target_zip.name} ({target_zip.stat().st_size / 1024:.2f} KB)")
        return True
    except Exception as e:
        logger.error(f"❌ ZIP creation failed: {e}")
        return False

def cleanup_old_backups() -> int:
    cutoff = datetime.now() - timedelta(days=config.RETENTION_DAYS)
    deleted = 0
    for folder in Path(config.BASE_DIR).glob("session_*"):
        try:
            folder_time = datetime.strptime(folder.name.split("_", 1)[1], "%Y-%m-%d_%H-%M")
            if folder_time < cutoff:
                shutil.rmtree(folder)
                logger.info(f"🗑️ Deleted old backup: {folder.name}")
                deleted += 1
        except Exception as e:
            logger.warning(f"⚠️ Could not delete {folder.name}: {e}")
    return deleted

def send_report(saved_files: int, zip_created: bool, deleted_backups: int):
    status = "✅ Backup erfolgreich" if saved_files else "⚠️ Backup fehlgeschlagen"
    msg = (
        f"*{status}*\n"
        f"`{datetime.now().strftime('%Y-%m-%d %H:%M')}`\n"
        f"📂 Dateien: `{saved_files}`\n"
        f"📦 ZIP: `{'Ja' if zip_created else 'Nein'}`\n"
        f"🗑️ Alte Backups gelöscht: `{deleted_backups}`"
    )
    send_telegram_message(msg, level="normal" if saved_files else "error")

def telegram_self_test():
    """Testet Telegram-Kommunikation vor Backup."""
    try:
        test_msg = "🛠️ *Telegram Verbindung OK*\n_CoreFlow-Backup beginnt jetzt..._"
        send_telegram_message(test_msg, level="normal")
        logger.info("📡 Telegram-Verbindung erfolgreich getestet.")
    except Exception as e:
        logger.warning(f"⚠️ Telegram-Test fehlgeschlagen: {e}")

def main():
    logger.info("🚀 Starte CoreFlow Backup")
    telegram_self_test()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    archive_dir = Path(config.BASE_DIR) / f"session_{timestamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    saved_files = backup_files(archive_dir)
    zip_created = zip_project(archive_dir / config.ZIP_NAME)
    deleted_backups = cleanup_old_backups()

    send_report(saved_files, zip_created, deleted_backups)
    logger.info("✅ Backup abgeschlossen")

if __name__ == "__main__":
    main()
