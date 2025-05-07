import base64
import os
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

def decrypt_env():
    with open(KEY_PATH, "rb") as f:
        key = f.read()
    fernet = Fernet(key)
    with open(ENCRYPTED_ENV_PATH, "rb") as f:
        decrypted = fernet.decrypt(f.read())
    with open(DECRYPTED_ENV_PATH, "wb") as f:
        f.write(decrypted)
    load_dotenv(DECRYPTED_ENV_PATH)
    print("🔓 .env entschlüsselt und geladen")

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else None
    if cmd == "genkey":
        generate_key()
    elif cmd == "encrypt":
        encrypt_env()
    elif cmd == "decrypt":
        decrypt_env()
    else:
        print("Verwendung: python env_crypto.py [genkey|encrypt|decrypt]")
