import os
import logging
from cryptography.fernet import Fernet

logging.basicConfig(level=logging.DEBUG)

def debug_decrypt():
    enc_path = "/opt/coreflow/.env.enc"
    key_path = "/opt/coreflow/infra/vault/encryption.key"
    
    try:
        # 1. Schlüsselprüfung
        logging.info(f"Prüfe Schlüsseldatei: {key_path}")
        if not os.path.exists(key_path):
            logging.error("❌ Schlüsseldatei nicht gefunden")
            return False
        
        # 2. Schlüssellänge prüfen
        with open(key_path, "rb") as f:
            key = f.read()
            logging.info(f"Schlüssellänge: {len(key)} Bytes")
            if len(key) != 44:  # Standard Fernet key length
                logging.error("❌ Ungültige Schlüssellänge")
        
        # 3. Verschlüsselte Datei prüfen
        logging.info(f"Prüfe verschlüsselte Datei: {enc_path}")
        if not os.path.exists(enc_path):
            logging.error("❌ .env.enc nicht gefunden")
            return False
        
        # 4. Entschlüsselung testen
        fernet = Fernet(key)
        with open(enc_path, "rb") as f:
            encrypted = f.read()
            logging.info(f"Verschlüsselte Datenlänge: {len(encrypted)} Bytes")
        
        decrypted = fernet.decrypt(encrypted).decode()
        logging.info("✅ Erfolgreich entschlüsselt")
        print("\nErste 50 Zeichen der entschlüsselten Daten:")
        print(decrypted[:50] + "...")
        
        return True
        
    except Exception as e:
        logging.exception("❌ Kritischer Fehler:")
        return False

if __name__ == "__main__":
    debug_decrypt()
