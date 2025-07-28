#!/usr/bin/env python3

import os
from pathlib import Path
from cryptography.fernet import Fernet

# === Standardpfade ===
paths = {
    "FERNET_KEY": Path("/opt/coreflow/infra/vault/encryption.key"),
    "ENCRYPTED_ENV": Path("/opt/coreflow/.env.enc"),
    "REDIS_AUTH_FILE": Path("/opt/coreflow/secrets/redis.pass"),
    "DECRYPTED_ENV_TMP": Path("/opt/coreflow/tmp/.env"),
    "LOG_DIR": Path("/opt/coreflow/logs"),
    "EMITTER_LOG": Path("/opt/coreflow/logs/signal_emitter.log"),
}

print("\n🔍 COREFLOW ENVIRONMENT CHECKER\n" + "-"*40)

# === Prüfen ob Pfade existieren ===
for name, path in paths.items():
    if path.exists():
        if path.is_dir():
            print(f"✅ {name} → Verzeichnis vorhanden: {path}")
        else:
            print(f"✅ {name} → Datei vorhanden: {path}")
    else:
        print(f"❌ {name} fehlt: {path}")

# === Entschlüsselungstest ===
if paths["FERNET_KEY"].exists() and paths["ENCRYPTED_ENV"].exists():
    try:
        key = Fernet(paths["FERNET_KEY"].read_bytes())
        decrypted = key.decrypt(paths["ENCRYPTED_ENV"].read_bytes()).decode()
        print(f"\n🔓 .env.enc erfolgreich entschlüsselt:")
        for line in decrypted.strip().splitlines():
            if line.strip() and not line.strip().startswith("#"):
                print("  " + line)
        # temporäre Datei schreiben (simulieren)
        paths["DECRYPTED_ENV_TMP"].parent.mkdir(parents=True, exist_ok=True)
        paths["DECRYPTED_ENV_TMP"].write_text(decrypted)
        print(f"\n✅ Temporäre entschlüsselte Datei geschrieben: {paths['DECRYPTED_ENV_TMP']}")
    except Exception as e:
        print(f"❌ Fehler beim Entschlüsseln von .env.enc: {e}")
else:
    print("⚠️ Entschlüsselung übersprungen – Schlüssel oder Datei fehlen.")

# === Redis Passwort prüfen ===
if paths["REDIS_AUTH_FILE"].exists():
    pw = paths["REDIS_AUTH_FILE"].read_text().strip()
    if pw:
        print(f"\n🔐 REDIS Passwort-Datei OK (Länge: {len(pw)} Zeichen)")
    else:
        print("❌ REDIS Passwort-Datei ist leer!")
else:
    print("❌ REDIS Passwort-Datei fehlt!")

# === Abschluss ===
print("\n✅ Prüfung abgeschlossen.")
