#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CoreFlow Secure Backup Script v1.0
🔐 Erstellt verschlüsselte Backups, sendet Statusmails, überträgt via SCP
"""

import os
import tarfile
import logging
import smtplib
import gnupg
from datetime import datetime
from email.mime.text import MIMEText
from subprocess import run, CalledProcessError

# === Konfiguration ===
BACKUP_DIRS = [
    "/opt/coreflow/core",
    "/opt/coreflow/config",
    "/opt/coreflow/secure",
    "/opt/coreflow/risk_management",
    "/opt/coreflow/service"
]

TMP_BACKUP_PATH = "/opt/coreflow/backups/coreflow_backup.tar.gz"
ENCRYPTED_BACKUP_PATH = "/opt/coreflow/backups/coreflow_backup.tar.gz.gpg"

GPG_RECIPIENT = "coreflow@ynuts.de"
SCP_DESTINATION = "coreadmin@os.getenv('REDIS_HOST'):/C:/Users/coreadmin/Backups/coreflow/"

SMTP_SERVER = "smtp.strato.de"
SMTP_PORT = 587
SMTP_USER = "coreflow@ynuts.de"
SMTP_PASS = "ne*L*io6610!?!"
SENDER = "coreflow@ynuts.de"
RECIPIENTS = ["coreflow@ynuts.de", "mabluco@mabluco.com"]

# === Logging ===
LOG_PATH = "/opt/coreflow/logs/backup.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)

# === Funktionen ===
def create_backup():
    logging.info("📦 Erstelle TAR-Backup ...")
    with tarfile.open(TMP_BACKUP_PATH, "w:gz") as tar:
        for path in BACKUP_DIRS:
            if os.path.exists(path):
                tar.add(path, arcname=os.path.basename(path))
            else:
                logging.warning(f"⚠️ Verzeichnis nicht gefunden: {path}")
    return TMP_BACKUP_PATH

def encrypt_backup(input_path, output_path):
    logging.info("🔐 Verschlüssle Backup mit GPG ...")
    gpg = gnupg.GPG()
    with open(input_path, 'rb') as f:
        status = gpg.encrypt_file(
            f,
            recipients=[GPG_RECIPIENT],
            output=output_path
        )
    if not status.ok:
        raise RuntimeError(f"GPG Fehler: {status.status}")
    logging.info("✅ Backup erfolgreich verschlüsselt")

def send_mail(subject, body):
    logging.info("📨 Sende Statusmail ...")
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = SENDER
    msg['To'] = ", ".join(RECIPIENTS)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SENDER, RECIPIENTS, msg.as_string())
        logging.info("✅ Statusmail gesendet")
    except Exception as e:
        logging.error(f"❌ Fehler beim Senden der Mail: {e}")

def upload_scp(file_path):
    logging.info(f"🚀 Sende Backup via SCP nach {SCP_DESTINATION} ...")
    try:
        result = run(["scp", file_path, SCP_DESTINATION], check=True)
        logging.info("✅ SCP-Upload abgeschlossen")
    except CalledProcessError as e:
        logging.error(f"❌ SCP-Fehler: {e}")

# === Hauptprozess ===
if __name__ == "__main__":
    logging.info("🚀 Starte CoreFlow Secure Backup ...")
    try:
        create_backup()
        encrypt_backup(TMP_BACKUP_PATH, ENCRYPTED_BACKUP_PATH)
        upload_scp(ENCRYPTED_BACKUP_PATH)
        send_mail(
            "✅ CoreFlow Backup erfolgreich",
            f"Das Backup wurde am {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} erfolgreich erstellt und hochgeladen."
        )
    except Exception as e:
        logging.critical(f"❌ Kritischer Fehler: {e}")
        send_mail(
            "❌ CoreFlow Backup fehlgeschlagen",
            f"Beim Backup ist ein Fehler aufgetreten:\n\n{e}"
        )
