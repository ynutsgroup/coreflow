#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoreFlow Auto-Backup Script (daily .tar.gz + cleanup)
"""

import os
import tarfile
from datetime import datetime, timedelta
import logging

# === Konfiguration ===
BACKUP_DIR = "/opt/coreflow/backups"
TARGET_DIR = "/opt/coreflow"
DAYS_TO_KEEP = 7
EXCLUDE_DIRS = ["backups", ".git", "__pycache__", "venv"]

# === Logging ===
LOG_FILE = os.path.join(BACKUP_DIR, "backup_log.txt")
os.makedirs(BACKUP_DIR, exist_ok=True)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s | %(message)s")

def should_exclude(name):
    return any(skip in name for skip in EXCLUDE_DIRS)

def create_backup():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{BACKUP_DIR}/coreflow_env_backup_{timestamp}.tar.gz"

    with tarfile.open(filename, "w:gz") as tar:
        for root, dirs, files in os.walk(TARGET_DIR):
            if should_exclude(root): continue
            for file in files:
                full_path = os.path.join(root, file)
                if should_exclude(full_path): continue
                arcname = os.path.relpath(full_path, TARGET_DIR)
                tar.add(full_path, arcname=arcname)
    
    logging.info(f"✅ Backup erstellt: {filename}")
    print(f"Backup gespeichert: {filename}")

def cleanup_old_backups():
    cutoff = datetime.now() - timedelta(days=DAYS_TO_KEEP)
    for file in os.listdir(BACKUP_DIR):
        if file.endswith(".tar.gz"):
            path = os.path.join(BACKUP_DIR, file)
            if os.path.isfile(path) and datetime.fromtimestamp(os.path.getmtime(path)) < cutoff:
                os.remove(path)
                logging.info(f"🧹 Alte Sicherung gelöscht: {file}")

if __name__ == "__main__":
    create_backup()
    cleanup_old_backups()
