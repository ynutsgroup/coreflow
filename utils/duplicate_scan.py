#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Duplicate Function Usage Scanner – CoreFlow Edition (safe mode)
Ignoriert .venv und schützt vor Abstürzen durch try/except
"""

import os
import ast
import pandas as pd
from collections import defaultdict

# === Globale Einstellungen ===
PROJECT_ROOT = "/opt/coreflow"
EXCLUDE_DIRS = [".venv", "__pycache__", "site-packages", "logs", "backup", ".git"]
VALID_EXTENSIONS = [".py"]

# === Duplikatspeicher initialisieren ===
function_usage = defaultdict(lambda: defaultdict(int))  # {file: {func_name: count}}

# === Dateien rekursiv scannen ===
def scan_python_files(base_path):
    for root, dirs, files in os.walk(base_path):
        # Verzeichnisse ignorieren
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for file in files:
            if any(file.endswith(ext) for ext in VALID_EXTENSIONS):
                yield os.path.join(root, file)

# === Funktionen zählen (sicher mit try/except)
def count_function_calls(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=filepath)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                else:
                    name = "<unknown>"
                function_usage[filepath][name] += 1

    except Exception as e:
        print(f"⚠️ Fehler in {filepath}: {e}")

# === Scan ausführen
for py_file in scan_python_files(PROJECT_ROOT):
    count_function_calls(py_file)

# === Ergebnis in DataFrame
rows = []
for file, funcs in function_usage.items():
    for func, count in funcs.items():
        if count > 1:
            rows.append({"file": file, "function": func, "count": count})

df = pd.DataFrame(rows).sort_values(by="count", ascending=False)

# === Ausgabe anzeigen
if not df.empty:
    print("🚨 Duplikate gefunden:")
    print(df.to_string(index=False))
else:
    print("✅ Keine Duplikate gefunden.")
