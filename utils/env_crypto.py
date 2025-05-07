#!/usr/bin/env python3
# env_crypto.py – Verschlüsselte .env-Verwaltung für CoreFlow

import base64
import os
import sys
from cryptography.fernet import Fernet
from dotenv import load_dotenv

KEY_PATH = "/opt/coreflow/.env.key"
ENCRYPTED_ENV_PATH = "/opt/coreflow/.env.enc"
DECRYPTED_ENV_PATH = "/opt/coreflow/.env.dec"

def generate_key():
    key = Fernet.generate_key()
    with open(KEY_PATH, "wb") as f:
        f.write(key)
    print("✅ Schlüssel generiert:", KEY_PATH)

def encrypt_env(input_path: str = ".env", output_path: str = ENCRYPTED_ENV_PATH):
    with open(KEY_PATH, "rb") as f:
        key = f.read()
    fernet = Fernet(key)
    with open(input_path, "rb") as f:
        encrypted = fernet.encrypt(f.read())
    with open(output_path, "wb") as f:
        f.write(encrypted)
    print("🔒 .env verschlüsselt:", output_path)

def decrypt_env(output_path: str = DECRYPTED_ENV_PATH):
    with open(KEY_PATH, "rb") as f:
        key = f.read()
    fernet = Fernet(key)
    with open(ENCRYPTED_ENV_PATH, "rb") as f:
        decrypted = fernet.decrypt(f.read())
    with open(output_path, "wb") as f:
        f.write(decrypted)
    print("🔓 .env entschlüsselt:", output_path)

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "genkey":
        generate_key()
    elif len(sys.argv) == 3 and sys.argv[1] == "encrypt":
        encrypt_env(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == "decrypt":
        decrypt_env()
    else:
        print("❗ Nutzung:")
        print("  genkey                      → Schlüssel generieren")
        print("  encrypt <pfad zur .env>    → .env verschlüsseln")
        print("  decrypt                    → .env entschlüsseln")
