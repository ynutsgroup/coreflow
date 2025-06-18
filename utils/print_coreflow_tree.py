import os

# Function to generate filtered tree
EXCLUDE_DIRS = {'.git', '__pycache__', '.venv', '.idea', 'checkpoints', '__model__'}
EXCLUDE_EXT = {'.log', '.pyc', '.tmp', '.bak'}

def is_excluded(name):
    if name in EXCLUDE_DIRS:
        return True
    _, ext = os.path.splitext(name)
    return ext in EXCLUDE_EXT

def write_tree(start_path, prefix=""):
    tree_lines = []
    try:
        entries = [e for e in sorted(os.listdir(start_path)) if not is_excluded(e)]
    except PermissionError:
        return tree_lines

    entries_count = len(entries)

    for i, entry in enumerate(entries):
        path = os.path.join(start_path, entry)
        connector = "└── " if i == entries_count - 1 else "├── "
        tree_lines.append(prefix + connector + entry)
        if os.path.isdir(path):
            extension = "    " if i == entries_count - 1 else "│   "
            tree_lines.extend(write_tree(path, prefix + extension))

    return tree_lines

# Base directory to scan
base_dir = "/opt/coreflow"
tree_output = []

if os.path.exists(base_dir):
    tree_output.append(f"📂 Struktur von {base_dir} (gefiltert):\n")
    tree_output.append(base_dir)
    tree_output.extend(write_tree(base_dir))
else:
    tree_output.append(f"❌ Verzeichnis {base_dir} existiert nicht.")

# Display output
import ace_tools as tools; tools.display_dataframe_to_user(name="CoreFlow Struktur (gefiltert)", dataframe={"Struktur": tree_output})
