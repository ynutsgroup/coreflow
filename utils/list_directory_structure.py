import os
import pwd
import sys
from pathlib import Path

def is_my_file(filepath):
    """Prüft, ob die Datei dem aktuellen Benutzer gehört"""
    try:
        return os.stat(filepath).st_uid == os.getuid()
    except:
        return False

def list_coreflow_structure(root_path):
    """Listet Dateien im Hauptverzeichnis und direkte Unterverzeichnisse auf"""
    valid_extensions = {'.py', '.sh', '.md', '.log', '.json', '.yaml', '.txt'}
    valid_filenames = {'.env', 'git_sync.py', 'requirements.txt'}
    
    root_path = Path(root_path)
    if not root_path.is_dir():
        return {}

    # Hauptverzeichnis
    root_files = []
    for f in os.listdir(root_path):
        filepath = root_path / f
        if filepath.is_file():
            if f.endswith(tuple(valid_extensions)) or f in valid_filenames:
                if is_my_file(filepath):
                    root_files.append(f)

    # Unterverzeichnisse (nur eine Ebene tief)
    subdirs = {}
    for d in os.listdir(root_path):
        dirpath = root_path / d
        if dirpath.is_dir():
            dir_files = []
            for f in os.listdir(dirpath):
                filepath = dirpath / f
                if filepath.is_file():
                    if f.endswith(tuple(valid_extensions)) or f in valid_filenames:
                        if is_my_file(filepath):
                            dir_files.append(f)
            if dir_files:
                subdirs[d] = sorted(dir_files)

    return {
        'root': sorted(root_files),
        'subdirs': subdirs
    }

def print_structure(structure):
    """Gibt die Verzeichnisstruktur aus"""
    if not structure['root'] and not structure['subdirs']:
        print("Keine passenden Dateien in /opt/coreflow gefunden")
        return
    
    print(f"\n📁 /opt/coreflow")
    
    # Dateien im Hauptverzeichnis
    for i, file in enumerate(structure['root']):
        prefix = "└──" if (i == len(structure['root'])-1 and not structure['subdirs']) else "├──"
        print(f"{prefix} 📄 {file}")
    
    # Unterverzeichnisse
    subdirs = structure['subdirs']
    for i, (dir_name, files) in enumerate(subdirs.items()):
        prefix = "└──" if i == len(subdirs)-1 else "├──"
        print(f"{prefix} 📁 {dir_name}/")
        
        for j, file in enumerate(files):
            sub_prefix = "    └──" if j == len(files)-1 else "    ├──"
            print(f"{sub_prefix} 📄 {file}")

if __name__ == "__main__":
    root_path = '/opt/coreflow'
    if not os.path.exists(root_path):
        print(f"Fehler: Pfad {root_path} existiert nicht!", file=sys.stderr)
        sys.exit(1)
    
    structure = list_coreflow_structure(root_path)
    print_structure(structure)
