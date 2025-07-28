#!/usr/bin/env python3
import json
import os

manifest_path = "/opt/coreflow/structure_manifest.json"
output_path = "/opt/coreflow/tree.txt"

# Baumzeichen
def print_tree(data):
    lines = []
    for dir_index, (folder, files) in enumerate(data.items()):
        folder_symbol = "└──" if dir_index == len(data)-1 else "├──"
        lines.append(f"{folder_symbol} {folder}")
        for file_index, f in enumerate(files):
            file_symbol = "└──" if file_index == len(files)-1 else "├──"
            lines.append(f"    {file_symbol} {f}")
    return lines

def main():
    if not os.path.exists(manifest_path):
        print(f"❌ Manifest nicht gefunden: {manifest_path}")
        return

    with open(manifest_path, "r") as f:
        data = json.load(f)

    lines = print_tree(data)

    with open(output_path, "w") as out:
        out.write("\n".join(lines) + "\n")

    print(f"✅ Strukturbaum gespeichert unter: {output_path}")

if __name__ == "__main__":
    main()
