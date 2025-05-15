import os
import time
import hashlib
import logging
import requests
from pathlib import Path
from filelock import FileLock
from dotenv import load_dotenv
from typing import Set, Optional, Tuple

# Konfiguration laden
load_dotenv(dotenv_path="/opt/coreflow/.env")

class UltimateMessageSender:
    """Endgültige Lösung mit allen Schutzmechanismen"""
    
    def __init__(self):
        # Basis-Konfiguration
        self.storage_path = Path("/opt/coreflow/msg_store.db")
        self.lock_path = Path("/opt/coreflow/msg_store.lock")
        self.sent_hashes: Set[str] = set()
        
        # Telegram Einstellungen
        self.token = os.getenv("TELEGRAM_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        
        # Schutzmechanismen
        self.min_interval = 2  # Mindestabstand in Sekunden
        self.last_send_time = 0
        self.max_retention_hours = 24  # Nachrichten nur 24h speichern
        
        # Initialisierung
        self._setup_storage()
    
    def _setup_storage(self):
        """Initialisiert die Nachrichtenspeicherung"""
        try:
            self.storage_path.parent.mkdir(exist_ok=True, parents=True)
            with FileLock(self.lock_path):
                if self.storage_path.exists():
                    with open(self.storage_path, 'r') as f:
                        self._clean_old_entries(f.readlines())
        except Exception as e:
            logging.error(f"Storage init failed: {str(e)}")
    
    def _clean_old_entries(self, entries: list):
        """Bereinigt alte Einträge"""
        cutoff = time.time() - (self.max_retention_hours * 3600)
        with open(self.storage_path, 'w') as f:
            for entry in entries:
                try:
                    timestamp, msg_hash = entry.strip().split('|')
                    if float(timestamp) > cutoff:
                        self.sent_hashes.add(msg_hash)
                        f.write(entry)
                except ValueError:
                    continue
    
    def _create_hash(self, message: str) -> str:
        """Erzeugt konsistenten Hash"""
        clean_msg = message.split(' - ')[0]  # Entfernt variable Teile
        return hashlib.sha256(clean_msg.encode()).hexdigest()
    
    def _send_telegram(self, message: str) -> bool:
        """Sichere Telegram-Sendefunktion"""
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "HTML"
                },
                timeout=15
            )
            return response.status_code == 200
        except Exception as e:
            logging.error(f"Telegram error: {str(e)}")
            return False
    
    def send_message(
        self,
        message: str,
        group: Optional[str] = None,
        force: bool = False
    ) -> Tuple[bool, str]:
        """
        Sendet Nachricht nur wenn einzigartig
        
        Returns:
            Tuple: (success, status_message)
        """
        # Erstelle konsistenten Hash
        unique_id = f"{group}:{message}" if group else message
        msg_hash = self._create_hash(unique_id)
        
        # Rate-Limiting
        elapsed = time.time() - self.last_send_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        
        # Duplikatprüfung
        with FileLock(self.lock_path):
            if not force and msg_hash in self.sent_hashes:
                return (False, "duplicate")
            
            # Sendeversuch
            if self._send_telegram(message):
                self.sent_hashes.add(msg_hash)
                with open(self.storage_path, 'a') as f:
                    f.write(f"{time.time()}|{msg_hash}\n")
                self.last_send_time = time.time()
                return (True, "sent")
            return (False, "send_failed")

# Initialisierung
message_sender = UltimateMessageSender()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/opt/coreflow/logs/message_sender.log'),
            logging.StreamHandler()
        ]
    )
    
    # Testcases
    results = []
    results.append(message_sender.send_message("System Start"))  # Sollte senden
    results.append(message_sender.send_message("System Start"))  # Sollte blockieren
    results.append(message_sender.send_message("System Start", force=True))  # Erzwingen
    
    logging.info(f"Test results: {results}")
