#!/usr/bin/env python3
# /opt/coreflow/utils/test_scp_env.py
# Prüft, ob SCP-Verbindung zum Windows-Server funktioniert und Dateien vorhanden sind

import subprocess
import datetime

# Konfiguration
windows_ip = "192.168.178.20"
windows_user = "Administrator"
remote_dir = "/C/CoreFlow"
remote_env = f"{windows_user}@{windows_ip}:{remote_dir}/.env"
remote_key = f"{windows_user}@{windows_ip}:{remote_dir}/.env.key"

local_env = "/opt/coreflow/.env"
local_key = "/opt/coreflow/.env.key"

logfile = "/opt/coreflow/logs/scp_env_test.log"

def log(msg):
    timestamp = datetime.datetime.utcnow().isoformat()
    with open(logfile, "a") as f:
        f.write(f"{timestamp} {msg}\n")
    print(msg)

def test_scp_file(remote_path, local_path):
    try:
        result = subprocess.run(
            ["scp", remote_path, local_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15
        )
        log(f"✅ Übertragen: {remote_path} → {local_path}")
    except subprocess.CalledProcessError as e:
        log(f"❌ SCP-Fehler: {remote_path} → {e.stderr.decode().strip()}")
    except subprocess.TimeoutExpired:
        log(f"⏱️ SCP-Zeitüberschreitung: {remote_path}")
    except Exception as e:
        log(f"❌ Unerwarteter Fehler: {e}")

if __name__ == "__main__":
    log("🔍 Starte SCP-Test für .env und .env.key")
    test_scp_file(remote_env, local_env)
    test_scp_file(remote_key, local_key)
