import os
from cryptography.fernet import Fernet

def decrypt_env(enc_path: str, key_path: str) -> dict:
    with open(key_path, 'rb') as f:
        key = f.read()
    fernet = Fernet(key)
    with open(enc_path, 'rb') as ef:
        decrypted = fernet.decrypt(ef.read()).decode()
    lines = decrypted.strip().split('\n')
    return {k: v for k, v in [line.strip().split('=', 1) for line in lines if '=' in line]}
