#!/usr/bin/env python3
# CoreFlow Utility – IP-Refactoring Script

import os

# Konfiguration – alte & neue IP hier definieren
OLD_IP = "10.10.10.1"
NEW_IP = "10.10.10.1"

# CoreFlow-Hauptverzeichnis
TARGET_DIR = "/opt/coreflow"

# Nur in diesen Dateitypen suchen/ersetzen
FILETYPES = (".py", ".json", ".yaml", ".yml", ".env", ".sh")

def update_ip_in_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if OLD_IP in content:
            new_content = content.replace(OLD_IP, NEW_IP)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"✔️ Updated: {file_path}")
    except Exception as e:
        print(f"❌ Fehler bei {file_path}: {e}")

def traverse_and_update(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(FILETYPES):
                update_ip_in_file(os.path.join(root, file))

if __name__ == "__main__":
    print(f"🔧 IP-Update: {OLD_IP} → {NEW_IP} in {TARGET_DIR}")
    traverse_and_update(TARGET_DIR)
    print("✅ Fertig.")
