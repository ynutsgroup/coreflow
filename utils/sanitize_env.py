#!/usr/bin/env python3

original_env_path = "/opt/coreflow/.env"
cleaned_env_path = "/opt/coreflow/.env.cleaned"

cleaned_lines = []
with open(original_env_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line and len(line.split("=", 1)) == 2:
            cleaned_lines.append(line)

with open(cleaned_env_path, "w", encoding="utf-8") as f:
    f.write("\n".join(cleaned_lines) + "\n")

print(f"✅ Gesäuberte Datei gespeichert als: {cleaned_env_path}")
