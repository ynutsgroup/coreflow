#!/usr/bin/env python3
import os
import subprocess
import logging
import sys
from datetime import datetime
from pathlib import Path

# Konfiguration
PROJECT_DIR = Path("/opt/coreflow")
LOG_FILE = PROJECT_DIR / "logs/git_sync.log"
GIT_EXCLUDES = [".env", ".env.key", "__pycache__", "*.log"]
GIT_REMOTE = "origin"  # Standard Remote-Repository
GIT_BRANCH = "main"    # Standard Branch

# Logging einrichten
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("GitSync")

def git_command(command, cwd=PROJECT_DIR, log_errors=True):
    """Führt Git-Befehle aus mit verbesserter Fehlerbehandlung."""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=30  # Timeout in Sekunden
        )
        output = result.stdout.strip()
        if output:  # Nur loggen wenn Ausgabe vorhanden
            logger.info(f"$ {' '.join(command)}\n{output}")
        return True, output
    except subprocess.TimeoutExpired:
        error_msg = f"Timeout bei: {' '.join(command)}"
        logger.error(error_msg)
        if log_errors:
            print(error_msg, file=sys.stderr)
        return False, error_msg
    except subprocess.CalledProcessError as e:
        error_msg = f"Fehler bei: {' '.join(command)}\n{e.stderr.strip()}"
        logger.error(error_msg)
        if log_errors:
            print(error_msg, file=sys.stderr)
        return False, error_msg

def get_changed_files():
    """Ermittelt geänderte Dateien unter Berücksichtigung der Ausschlussliste."""
    try:
        # Nur Dateien die bereits im Git-Index sind
        tracked_files = subprocess.check_output(
            ["git", "ls-files"],
            cwd=PROJECT_DIR,
            text=True
        ).splitlines()
        
        # Geänderte Dateien
        modified = subprocess.check_output(
            ["git", "diff", "--name-only", "--diff-filter=M"],
            cwd=PROJECT_DIR,
            text=True
        ).splitlines()
        
        # Neue Dateien
        added = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=PROJECT_DIR,
            text=True
        ).splitlines()
        
        # Kombinieren und ausschließen
        all_changes = set(modified + added)
        return [f for f in all_changes if not any(
            exclude in f for exclude in GIT_EXCLUDES
        )]
    except subprocess.CalledProcessError as e:
        logger.error(f"Fehler beim Ermitteln der Änderungen: {e.stderr.strip()}")
        return []

def sync_git():
    """Hauptfunktion für die Git-Synchronisation."""
    logger.info("🚀 Git Sync gestartet")
    
    try:
        os.chdir(PROJECT_DIR)
        
        # Aktuellen Branch ermitteln
        success, branch_output = git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], log_errors=False)
        if not success:
            branch_output = GIT_BRANCH

        # Remote Updates holen
        logger.info(f"Aktualisiere von {GIT_REMOTE}/{branch_output.strip()}")
        git_command(["git", "fetch", GIT_REMOTE])
        
        # Lokale Änderungen zurücksichern (Stash)
        git_command(["git", "stash", "push", "--include-untracked"])
        
        # Rebase durchführen
        git_command(["git", "rebase", f"{GIT_REMOTE}/{branch_output.strip()}"])
        
        # Stash wieder anwenden falls vorhanden
        git_command(["git", "stash", "pop"])
        
        # Änderungen ermitteln
        changes = get_changed_files()
        
        if changes:
            logger.info(f"🔄 {len(changes)} Änderungen gefunden")
            git_command(["git", "add", "."])
            
            commit_message = f"🔄 Auto-Sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            git_command(["git", "commit", "-m", commit_message])
            
            git_command(["git", "push", GIT_REMOTE, branch_output.strip()])
            logger.info("✅ Git Sync erfolgreich abgeschlossen")
        else:
            logger.info("🟢 Keine Änderungen zum Synchronisieren gefunden")
            
    except Exception as e:
        logger.critical(f"🔥 Kritischer Fehler: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    if sync_git():
        sys.exit(0)
    else:
        sys.exit(1)
