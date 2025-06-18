#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
encrypt_env.py – Verschlüsselt eine .env-Datei mit Fernet
Speichert Ergebnis als .env.enc für sichere Nutzung durch decrypt_env.py
"""

import os
from cryptography.fernet import Fernet
import logging

# === Logging (Konsolenausgabe) ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("EnvEncryptor")

# === Pfade ===
ENV_PATH = "/opt/coreflow/.env"
ENC_PATH = "/opt/coreflow/.env.enc"
KEY_PATH = "/opt/coreflow/infra/vault/encryption.key"

# === Schlüssel erzeugen (einmalig) ===
def generate_key(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        logger.info(f"🔐 Schlüssel existiert bereits: {path}")
        return
    key = Fernet.generate_key()
    with open(path, "wb") as f:
        f.write(key)
    logger.info(f"✅ Neuer Fernet-Schlüssel gespeichert: {path}")

# === .env verschlüsseln ===
def encrypt_env(env_path: str, key_path: str, out_path: str):
    if not os.path.exists(env_path):
        logger.critical(f"❌ .env-Datei nicht gefunden: {env_path}")
        return

    try:
        with open(key_path, "rb") as f:
            key = f.read()
        fernet = Fernet(key)

        with open(env_path, "rb") as f:
            plaintext = f.read()

        encrypted = fernet.encrypt(plaintext)

        with open(out_path, "wb") as f:
            f.write(encrypted)

        logger.info(f"✅ .env erfolgreich verschlüsselt → {out_path}")

    except Exception as e:
        logger.critical(f"❌ Verschlüsselung fehlgeschlagen: {e}")

# === Main ===
if __name__ == "__main__":
    generate_key(KEY_PATH)
    encrypt_env(ENV_PATH, KEY_PATH, ENC_PATH)
