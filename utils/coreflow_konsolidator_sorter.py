#!/usr/bin/env python3
import os, json, shutil, hashlib
from pathlib import Path
from datetime import datetime

# Konfiguration
ARCHIVE_DIR = "/opt/coreflow/archive/"
LOG_DIR = "/opt/coreflow/backup/"
DRY_RUN = False  # Auf True setzen für Testlauf ohne Änderungen
VERIFY_COPY = True  # Kopien verifizieren

def setup_dirs():
    """Sicherstellen, dass Verzeichnisse existieren mit korrekten Berechtigungen"""
    for dir_path in [ARCHIVE_DIR, LOG_DIR]:
        os.makedirs(dir_path, exist_ok=True)
        os.chmod(dir_path, 0o750)  # Sicherere Berechtigungen

def verify_copy(source, target):
    """Überprüft, ob Kopie identisch ist"""
    def hash_file(path):
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    return hash_file(source) == hash_file(target)

def find_latest_report():
    """Findet den aktuellsten Duplikat-Report mit Validierung"""
    backup_dir = Path(LOG_DIR)
    reports = sorted(backup_dir.glob("duplicate_report_*.json"), 
                   key=os.path.getmtime, reverse=True)
    
    if not reports:
        return None
        
    # Validierung des Reports
    latest = reports[0]
    try:
        with open(latest, 'r') as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Ungültiges Reportformat")
        return str(latest)
    except Exception as e:
        print(f"⚠️ Ungültiger Report {latest}: {e}")
        return None

def get_best_file(file_paths):
    """Wählt die beste Dateiversion basierend auf mehreren Kriterien"""
    candidates = []
    for path in file_paths:
        try:
            path = Path(path)
            stat = path.stat()
            candidates.append({
                'path': str(path),
                'size': stat.st_size,
                'mtime': stat.st_mtime,
                'depth': len(path.parts),  # Bevorzuge flachere Hierarchie
                'special': any(p.startswith('.') for p in path.parts)  # Versteckte Dateien nachrangig
            })
        except Exception as e:
            print(f"⚠️ Konnte {path} nicht analysieren: {e}")
            continue
    
    if not candidates:
        return None
    
    # Sortierpriorität: Größe > Aktualität > Hierarchietiefe > keine versteckten Pfade
    return sorted(candidates, 
                key=lambda x: (-x['size'], -x['mtime'], x['depth'], x['special']))[0]['path']

def archive_file(source, target_base):
    """Archiviert eine Datei mit Kollisionsbehandlung"""
    source_path = Path(source)
    target = Path(target_base) / source_path.name
    
    # Bei Namenskollisionen
    counter = 1
    while target.exists():
        target = Path(target_base) / f"{source_path.stem}_{counter}{source_path.suffix}"
        counter += 1
    
    if DRY_RUN:
        print(f"🔹 (DRY RUN) Würde verschieben: {source} → {target}")
        return str(target)
    
    try:
        # Sicherere Methode: Erst kopieren, dann verifizieren, dann original löschen
        temp_target = f"{target}.tmp"
        shutil.copy2(source, temp_target)
        
        if VERIFY_COPY and not verify_copy(source, temp_target):
            raise RuntimeError("Kopierverifizierung fehlgeschlagen")
            
        os.rename(temp_target, target)
        os.remove(source)
        return str(target)
    except Exception as e:
        print(f"❌ Kritischer Fehler bei {source}: {e}")
        if os.path.exists(temp_target):
            os.remove(temp_target)
        return None

def main():
    setup_dirs()
    
    duplicate_report = find_latest_report()
    if not duplicate_report:
        print("❌ Kein gültiger Duplikatsreport gefunden")
        exit(1)

    try:
        with open(duplicate_report, 'r', encoding='utf-8') as f:
            duplicates = json.load(f)
    except Exception as e:
        print(f"❌ Fehler beim Lesen des Reports: {e}")
        exit(1)

    moved = []
    stats = {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0}

    for group in duplicates:
        files = group.get('files', [])
        if len(files) < 2:
            continue
            
        stats['total'] += len(files) - 1  # Eine Datei pro Gruppe bleibt
        best = get_best_file(files)
        
        if not best:
            print(f"⚠️ Keine gültige Datei in Gruppe: {group.get('sha256', '?')}")
            stats['skipped'] += len(files)
            continue
            
        for f in files:
            if f == best:
                continue
                
            result = archive_file(f, ARCHIVE_DIR)
            if result:
                moved.append({'original': f, 'archived': result, 'sha256': group.get('sha256')})
                stats['success'] += 1
                print(f"🔁 {f} → {result}")
            else:
                stats['failed'] += 1
                print(f"⚠️ Fehler beim Archivieren von {f}")

    # Logs speichern
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_json = os.path.join(LOG_DIR, f"migration_log_{timestamp}.json")
    log_txt = os.path.join(LOG_DIR, f"migration_log_{timestamp}.txt")
    
    try:
        with open(log_json, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'timestamp': timestamp,
                    'report_source': duplicate_report,
                    'settings': {
                        'dry_run': DRY_RUN,
                        'verify_copy': VERIFY_COPY
                    }
                },
                'stats': stats,
                'moved_files': moved
            }, f, indent=2)
            
        with open(log_txt, 'w', encoding='utf-8') as f:
            f.write(f"Migration Report {timestamp}\n")
            f.write(f"Quellreport: {duplicate_report}\n")
            f.write(f"\nStatistiken:\n")
            f.write(f"- Total zu archivierende Dateien: {stats['total']}\n")
            f.write(f"- Erfolgreich: {stats['success']}\n")
            f.write(f"- Fehlgeschlagen: {stats['failed']}\n")
            f.write(f"- Übersprungen: {stats['skipped']}\n")
            f.write(f"\nDetails:\n")
            for item in moved:
                f.write(f"{item['original']} → {item['archived']} (SHA256: {item.get('sha256','?')})\n")
                
        print(f"\n✅ Migration abgeschlossen. Statistiken:")
        print(f"- Total: {stats['total']} | Erfolg: {stats['success']} | Fehler: {stats['failed']} | Übersprungen: {stats['skipped']}")
        print(f"📄 Logs gespeichert:\n→ {log_json}\n→ {log_txt}")
        
    except Exception as e:
        print(f"❌ Fehler beim Schreiben der Logs: {e}")
        exit(1)

if __name__ == "__main__":
    main()
