#!/usr/bin/env python3
from cryptography.fernet import Fernet

KEY_PATH = "/opt/coreflow/infra/vault/encryption.key"
ENV_PATH = "/opt/coreflow/.env"
ENC_PATH = "/opt/coreflow/.env.enc"

with open(KEY_PATH, "rb") as kf:
    key = kf.read()
fernet = Fernet(key)

with open(ENV_PATH, "rb") as f:
    encrypted = fernet.encrypt(f.read())

with open(ENC_PATH, "wb") as ef:
    ef.write(encrypted)

print("✅ .env erfolgreich verschlüsselt → .env.enc")
