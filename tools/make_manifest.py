#!/usr/bin/env python3
import os
import json

base_dir = "/opt/coreflow"
target_file = os.path.join(base_dir, "structure_manifest.json")
dirs = ["src", "logs", ".env", "notes"]

manifest = {}

for d in dirs:
    path = os.path.join(base_dir, d)
    if os.path.isdir(path):
        files = sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
        manifest[f"{d}/"] = files
    elif os.path.isfile(path):
        # Wenn es eine Datei ist, z.B. .env, dann als Liste mit einem Element speichern
        manifest[d] = [os.path.basename(path)]
    else:
        manifest[f"{d}/"] = []

with open(target_file, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"✅ Manifest mit valide JSON erstellt: {target_file}")
