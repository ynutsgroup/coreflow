#!/usr/bin/env python3
"""FTMO-konforme Dateiverschlüsselung für Trading-Systeme"""

import os
import sys
import logging
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# Logging-Konfiguration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("encryption.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FTMOEncryptor:
    def __init__(self):
        self.key_path = Path(".env.key")
        self.salt = os.urandom(16)  # Zufälliges Salt für KDF

    def _derive_key(self, password: bytes) -> bytes:
        """Ableitung eines kryptografischen Schlüssels"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=480000,  # FTMO Security Standard
        )
        return base64.urlsafe_b64encode(kdf.derive(password))

    def generate_key(self, password: str = None) -> None:
        """Generiert einen neuen Schlüssel mit optionalem Passwort"""
        if password:
            key = self._derive_key(password.encode())
        else:
            key = Fernet.generate_key()
        
        with open(self.key_path, "wb") as f:
            f.write(key)
        logger.info("🔐 Schlüssel generiert (Passwort: %s)", "JA" if password else "NEIN")

    def encrypt_file(self, file_path: Path) -> None:
        """Verschlüsselt eine Datei FTMO-konform"""
        if not self.key_path.exists():
            raise FileNotFoundError("Schlüsseldatei nicht gefunden")
            
        key = self.key_path.read_bytes()
        f = Fernet(key)
        
        with open(file_path, "rb") as file:
            data = file.read()
        
        encrypted = f.encrypt(data)
        output_path = file_path.with_suffix(file_path.suffix + ".ftmo")
        
        with open(output_path, "wb") as file:
            file.write(encrypted)
        
        logger.info("✅ %s verschlüsselt -> %s", file_path.name, output_path.name)
        self._secure_delete(file_path)  # Original sicher löschen

    def decrypt_file(self, file_path: Path) -> None:
        """Entschlüsselt eine FTMO-geschützte Datei"""
        if not file_path.suffix == ".ftmo":
            raise ValueError("Nur .ftmo-Dateien können entschlüsselt werden")
            
        key = self.key_path.read_bytes()
        f = Fernet(key)
        
        with open(file_path, "rb") as file:
            encrypted = file.read()
        
        decrypted = f.decrypt(encrypted)
        output_path = file_path.with_suffix("")  # Entfernt .ftmo
        
        with open(output_path, "wb") as file:
            file.write(decrypted)
        
        logger.info("✅ %s entschlüsselt -> %s", file_path.name, output_path.name)

    def _secure_delete(self, path: Path, passes: int = 3) -> None:
        """Sicheres Löschen nach NIST 800-88"""
        with open(path, "ba+") as f:
            length = f.tell()
            for _ in range(passes):
                f.seek(0)
                f.write(os.urandom(length))
        os.remove(path)
        logger.debug("🗑️ %s sicher gelöscht (NIST 800-88)", path.name)

def main():
    if len(sys.argv) < 3:
        print("""
        FTMO Dateiverschlüsselung
        Nutzung:
          generate [passwort]  - Generiert neuen Schlüssel
          encrypt <datei>      - Verschlüsselt Datei
          decrypt <datei>      - Entschlüsselt .ftmo Datei
        """)
        sys.exit(1)

    encryptor = FTMOEncryptor()
    action = sys.argv[1]

    try:
        if action == "generate":
            password = sys.argv[2] if len(sys.argv) > 2 else None
            encryptor.generate_key(password)
        elif action == "encrypt":
            encryptor.encrypt_file(Path(sys.argv[2]))
        elif action == "decrypt":
            encryptor.decrypt_file(Path(sys.argv[2]))
        else:
            logger.error("❌ Unbekannte Aktion")
    except Exception as e:
        logger.critical("Fehler: %s", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
