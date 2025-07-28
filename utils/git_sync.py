#!/usr/bin/env python3
import os
import subprocess
import logging
import sys
from datetime import datetime
from pathlib import Path

# === Konfiguration ===
PROJECT_DIR = Path("/opt/coreflow")
LOG_FILE = PROJECT_DIR / "logs/git_sync.log"
GIT_EXCLUDES = [".env", ".env.key", "__pycache__", "*.log"]
GIT_REMOTE = "origin"
GIT_BRANCH = "main"

# === Logging ===
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
LOG_FILE.touch(exist_ok=True)
LOG_FILE.chmod(0o664)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("GitSync")

def git_command(command, cwd=PROJECT_DIR, log_errors=True):
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        output = result.stdout.strip()
        if output:
            logger.info(f"$ {' '.join(command)}\n{output}")
        return True, output
    except subprocess.TimeoutExpired:
        error_msg = f"❌ Timeout: {' '.join(command)}"
        logger.error(error_msg)
        if log_errors:
            print(error_msg, file=sys.stderr)
        return False, error_msg
    except subprocess.CalledProcessError as e:
        error_msg = f"❌ Fehler: {' '.join(command)}\n{e.stderr.strip()}"
        logger.error(error_msg)
        if log_errors:
            print(error_msg, file=sys.stderr)
        return False, error_msg

def get_changed_files():
    try:
        tracked = subprocess.check_output(["git", "ls-files"], cwd=PROJECT_DIR, text=True).splitlines()
        modified = subprocess.check_output(["git", "diff", "--name-only", "--diff-filter=M"], cwd=PROJECT_DIR, text=True).splitlines()
        added = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=PROJECT_DIR, text=True).splitlines()
        all_changes = set(modified + added)
        return [f for f in all_changes if not any(ex in f for ex in GIT_EXCLUDES)]
    except subprocess.CalledProcessError as e:
        logger.error(f"Fehler beim Lesen der Änderungen: {e.stderr.strip()}")
        return []

def sync_git():
    logger.info("🚀 Git Sync gestartet")
    try:
        os.chdir(PROJECT_DIR)

        success, branch_output = git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], log_errors=False)
        if not success:
            branch_output = GIT_BRANCH
        else:
            branch_output = branch_output.strip()

        git_command(["git", "fetch", GIT_REMOTE])
        git_command(["git", "stash", "push", "--include-untracked"], log_errors=False)
        git_command(["git", "rebase", f"{GIT_REMOTE}/{branch_output}"], log_errors=False)
        git_command(["git", "stash", "pop"], log_errors=False)  # Kein Fehler wenn kein Stash

        changes = get_changed_files()
        if changes:
            logger.info(f"🔄 Änderungen gefunden: {len(changes)} Dateien")
            git_command(["git", "add", "."])

            success, staged = git_command(["git", "diff", "--cached", "--name-only"], log_errors=False)
            if success and staged.strip():
                commit_msg = f"🔄 Auto-Sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                git_command(["git", "commit", "-m", commit_msg])
                git_command(["git", "push", GIT_REMOTE, branch_output])
                logger.info("✅ Git Sync abgeschlossen")
            else:
                logger.info("ℹ️ Nichts zum Committen – nur lokale Änderungen")
        else:
            logger.info("🟢 Keine Änderungen entdeckt")

    except Exception as e:
        logger.critical(f"🔥 Kritischer Fehler: {str(e)}")
        return False
    return True

if __name__ == "__main__":
    if sync_git():
        sys.exit(0)
    else:
        sys.exit(1)
