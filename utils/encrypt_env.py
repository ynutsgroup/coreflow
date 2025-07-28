#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔐 Encrypts .env to .env.enc using Fernet
"""

import os
from cryptography.fernet import Fernet
from pathlib import Path

# === Pfade definieren ===
env_path = Path("/opt/coreflow/.env")
enc_path = Path("/opt/coreflow/.env.enc")
key_path = Path("/opt/coreflow/infra/vault/encryption.key")

# === Schlüssel laden ===
if not key_path.exists():
    print(f"❌ Key-Datei nicht gefunden: {key_path}")
    exit(1)

key = key_path.read_bytes()
fernet = Fernet(key)

# === .env laden und verschlüsseln ===
if not env_path.exists():
    print(f"❌ .env-Datei fehlt: {env_path}")
    exit(1)

plain = env_path.read_bytes()
encrypted = fernet.encrypt(plain)

# === .env.enc speichern ===
enc_path.write_bytes(encrypted)
print(f"✅ Verschlüsselt: {enc_path} ({enc_path.stat().st_size} bytes)")
