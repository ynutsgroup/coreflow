#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# CoreFlow Bridge Health Checker (Enhanced Version)

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

# Konfigurierbare Pfade
VENV_PYTHON = Path("/opt/coreflow/.venv/bin/python3")
BRIDGE_SCRIPT = Path("/opt/coreflow/bridge/redis_to_zmq_bridge.py")
SERVICE_FILE = Path("/etc/systemd/system/coreflow-bridge.service")
LOG_FILE = Path("/opt/coreflow/logs/redis_to_zmq_bridge.log")
ENV_FILE = Path("/opt/coreflow/.env")

# Required environment variables
REQUIRED_ENV_VARS = ["REDIS_HOST", "ZMQ_ADDR", "REDIS_PORT"]

def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n\033[1m{title}\033[0m")
    print("-" * len(title))

def check_path(
    path: Path,
    description: str,
    must_exist: bool = True,
    must_be_executable: bool = False,
) -> bool:
    """Check if a path exists and has required permissions."""
    if not path.exists():
        print(f"❌ {description} fehlt: {path}")
        return False
    if must_be_executable and not os.access(path, os.X_OK):
        print(f"❌ {description} ist nicht ausführbar: {path}")
        return False
    print(f"✅ {description} OK: {path}")
    return True

def check_service_status() -> Tuple[bool, str]:
    """Check systemd service status and return (is_active, is_enabled)."""
    try:
        active = subprocess.run(
            ["systemctl", "is-active", "coreflow-bridge.service"],
            capture_output=True,
            text=True,
        )
        enabled = subprocess.run(
            ["systemctl", "is-enabled", "coreflow-bridge.service"],
            capture_output=True,
            text=True,
        )
        is_active = active.stdout.strip() == "active"
        is_enabled = enabled.stdout.strip() == "enabled"
        
        status = f"Status: {'Aktiv' if is_active else 'Inaktiv'}, Autostart: {'Ja' if is_enabled else 'Nein'}"
        
        if is_active:
            print(f"✅ Service läuft: {status}")
        else:
            print(f"⚠️  Service nicht aktiv: {status}")
        
        return (is_active, is_enabled)
    except subprocess.CalledProcessError as e:
        print(f"❌ Fehler beim Service-Check: {e}")
        return (False, False)

def check_python_dependencies() -> bool:
    """Check if required Python packages are installed."""
    print_header("Python-Abhängigkeiten")
    try:
        result = subprocess.run(
            [str(VENV_PYTHON), "-m", "pip", "freeze"],
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )
        installed = result.stdout.lower()
        all_ok = True
        
        for pkg in ["redis", "pyzmq", "python-dotenv"]:
            if pkg in installed:
                print(f"✅ {pkg} installiert")
            else:
                print(f"❌ {pkg} fehlt – installieren mit:")
                print(f"   {VENV_PYTHON} -m pip install {pkg}")
                all_ok = False
        return all_ok
    except subprocess.CalledProcessError as e:
        print(f"❌ Pip-Abfrage fehlgeschlagen: {e}")
        return False

def check_env_vars() -> bool:
    """Verify required environment variables exist in .env file."""
    print_header(".env Prüfung")
    if not ENV_FILE.exists():
        print("❌ .env-Datei fehlt!")
        return False
    
    try:
        env_content = ENV_FILE.read_text()
        missing = [var for var in REQUIRED_ENV_VARS if f"{var}=" not in env_content]
        
        if missing:
            print(f"❌ Fehlende Variablen: {', '.join(missing)}")
            print("   Bitte ergänzen Sie diese in der .env-Datei.")
            return False
        
        print("✅ Alle benötigten Variablen vorhanden")
        return True
    except Exception as e:
        print(f"❌ Fehler beim Lesen der .env-Datei: {e}")
        return False

def check_logs() -> bool:
    """Check log files for errors and show recent entries."""
    print_header("Logfile-Analyse")
    if not LOG_FILE.exists():
        print("⚠️  Logdatei nicht gefunden. Wurde der Service schon gestartet?")
        return False
    
    try:
        # Show last 15 lines
        print("\nLetzte Log-Einträge:")
        subprocess.run(["tail", "-n", "15", str(LOG_FILE)], check=True)
        
        # Check for errors
        grep_result = subprocess.run(
            ["grep", "-i", "error|exception|fail", str(LOG_FILE)],
            capture_output=True,
            text=True,
        )
        
        if grep_result.stdout:
            print("\n❌ Fehler in Logs gefunden:")
            print(grep_result.stdout)
            return False
        
        print("\n✅ Keine Fehler in den Logs gefunden")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Log-Check fehlgeschlagen: {e}")
        return False

def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and handle errors gracefully."""
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {description} erfolgreich")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} fehlgeschlagen: {e}")
        return False

def suggest_fixes(issues: List[str]) -> None:
    """Provide actionable suggestions for detected issues."""
    if not issues:
        return
    
    print_header("Empfohlene Maßnahmen")
    for issue in issues:
        if "Service nicht aktiv" in issue:
            print("- Service starten:")
            print("  sudo systemctl start coreflow-bridge.service")
            print("- Aktivieren für Autostart:")
            print("  sudo systemctl enable coreflow-bridge.service")
        elif "Python-Abhängigkeiten fehlen" in issue:
            print("- Fehlende Pakete installieren:")
            print(f"  {VENV_PYTHON} -m pip install redis pyzmq python-dotenv")
        elif ".env fehlt" in issue:
            print("- .env-Datei erstellen und konfigurieren:")
            print("  cp /opt/coreflow/.env.example /opt/coreflow/.env")
            print("  nano /opt/coreflow/.env")

def main():
    print("\033[1m🔍 CoreFlow Bridge Health Check (v2.0)\033[0m")
    
    # Run all checks
    checks = {
        "Python-Pfad": check_path(VENV_PYTHON, "Python im venv", must_be_executable=True),
        "Bridge-Skript": check_path(BRIDGE_SCRIPT, "Bridge-Skript"),
        "Service-Datei": check_path(SERVICE_FILE, "Systemd Service-Datei"),
        "Env-Datei": check_env_vars(),
        "Python-Pakete": check_python_dependencies(),
        "Service-Status": check_service_status()[0],
        "Logs": check_logs(),
    }
    
    # Collect issues
    issues = [
        name for name, passed in checks.items() if not passed
    ]
    
    # Summary
    print_header("Zusammenfassung")
    if not issues:
        print("✅ Alles in Ordnung! Die Bridge sollte korrekt funktionieren.")
    else:
        print(f"❌ Probleme gefunden in: {', '.join(issues)}")
        suggest_fixes(issues)
    
    # Show systemd status at the end
    print_header("Systemd-Details")
    subprocess.run(["systemctl", "status", "coreflow-bridge.service", "--no-pager"])

if __name__ == "__main__":
    main()
