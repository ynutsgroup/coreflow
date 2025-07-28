import os
from datetime import datetime
from pathlib import Path

BASE_DIR = "/opt/coreflow_memory"
PROJECT_DIR = "/opt/coreflow"
SUMMARY_FILE = "chat_today.md"

def get_latest_session():
    sessions = sorted(Path(BASE_DIR).glob("session_*"))
    return sessions[-1] if sessions else None

def write_summary(session_dir: Path):
    timestamp = session_dir.name.replace("session_", "")
    structure_file = session_dir / "structure_manifest.json"
    progress_file = session_dir / "coreflow_progress.log"
    zip_file = session_dir / "autotrader_snapshot.zip"

    summary_path = Path(PROJECT_DIR) / SUMMARY_FILE
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# 📆 Session-Zusammenfassung – {timestamp}\n\n")

        f.write("## ✅ Gesicherte Dateien:\n")
        for file in [structure_file, progress_file, zip_file]:
            status = "✅" if file.exists() else "⚠️ Nicht gefunden"
            size = f"{file.stat().st_size / 1024:.1f} KB" if file.exists() else "-"
            f.write(f"- {file.name}: {status} ({size})\n")

        f.write("\n## 🧠 Projektstruktur:\n")
        f.write("- .env vorhanden ✅\n" if (Path(PROJECT_DIR) / ".env").exists() else "- .env fehlt ❌\n")
        log_dir = Path(PROJECT_DIR) / "logs"
        f.write(f"- logs/: {len(list(log_dir.glob('*.log')))} Dateien\n" if log_dir.exists() else "- logs/ fehlt ❌\n")
        notes_dir = Path(PROJECT_DIR) / "notes"
        f.write(f"- Notizen: {', '.join([p.name for p in notes_dir.glob('*.md')])}\n" if notes_dir.exists() else "- Notizen fehlen ❌\n")
        src_dir = Path(PROJECT_DIR) / "src"
        f.write(f"- src/: {'nicht leer ✅' if any(src_dir.rglob('*.py')) else 'leer ⚠️'}\n" if src_dir.exists() else "- src/ fehlt ❌\n")

        f.write("\n## 🔄 Wiederherstellung morgen:\n")
        f.write(f"```bash\npython /opt/coreflow/tools/coreflow_restore.py --session {timestamp} --full\n```\n")

    print(f"✅ Zusammenfassung geschrieben nach: {summary_path}")

if __name__ == "__main__":
    latest = get_latest_session()
    if latest:
        write_summary(latest)
    else:
        print("⚠️ Keine Session gefunden.")
