#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decrypt_env.py – Entschlüsselt eine Fernet-verschlüsselte .env-Datei (.env.enc)
und lädt sie direkt in den aktuellen Prozessspeicher
"""

import os
from dotenv import load_dotenv
from cryptography.fernet import Fernet
import logging

logger = logging.getLogger("EnvDecryptor")

def load_encrypted_env(enc_path="/opt/coreflow/.env.enc", key_path="/opt/coreflow/infra/vault/encryption.key") -> bool:
    try:
        with open(key_path, 'rb') as key_file:
            key = key_file.read()
        fernet = Fernet(key)

        with open(enc_path, 'rb') as enc_file:
            encrypted = enc_file.read()

        decrypted = fernet.decrypt(encrypted)

        temp_env_path = "/tmp/.coreflow_decrypted.env"
        with open(temp_env_path, 'wb') as temp_file:
            temp_file.write(decrypted)

        load_dotenv(dotenv_path=temp_env_path, override=True)
        logger.info("🔓 .env.enc erfolgreich entschlüsselt und geladen")
        return True

    except Exception as e:
        logger.critical(f"❌ Fehler beim Entschlüsseln der .env.enc: {e}")
        return False
