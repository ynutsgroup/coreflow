import os
from cryptography.fernet import Fernet
from pathlib import Path
from datetime import datetime

DEFAULT_KEY_PATH = "/opt/coreflow/infra/vault/encryption.key"

def load_env(env_path: str, key_path: str = DEFAULT_KEY_PATH) -> dict:
    try:
        with open(key_path, 'rb') as key_file:
            key = key_file.read()
        fernet = Fernet(key)

        with open(env_path, 'rb') as enc_file:
            encrypted = enc_file.read()
        decrypted = fernet.decrypt(encrypted).decode()

        env_dict = {}
        for line in decrypted.splitlines():
            if line.strip() and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()
                env_dict[k.strip()] = v.strip()
        return env_dict
    except Exception as e:
        print(f"❌ Fehler beim Entschlüsseln von {env_path}: {e}")
        return {}

def find_latest_env_enc(root_dir: str = "/opt/coreflow") -> str:
    paths = list(Path(root_dir).rglob(".env.enc"))
    if not paths:
        return None
    latest = max(paths, key=lambda p: p.stat().st_mtime)
    return str(latest)

def test_all_envs(root_dir: str = "/opt/coreflow"):
    print(f"🔍 Scanne nach .env.enc Dateien in {root_dir}")
    paths = sorted(Path(root_dir).rglob(".env.enc"))
    valid = False
    for p in paths:
        mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"➡ Prüfe: {p} (Letzte Änderung: {mtime})")
        try:
            env = load_env(str(p))
            if env:
                print(f"✅ OK: {p}")
                valid = True
        except Exception:
            print(f"❌ Fehler beim Entschlüsseln: {p}")
    if not valid:
        print("❌ Keine gültige .env.enc-Datei gefunden.")
