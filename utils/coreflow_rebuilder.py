#!/usr/bin/env python3
import os, json
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path("/opt/coreflow")

CATEGORIES = {
    "core": ["coreflow_main.py", "valg_engine.py", "signal_receiver.py"],
    "ai": ["lstm_predictor.py", "model_trainer.py", ".pkl", ".onnx"],
    "modules/timeseries": ["ts_aggregator.py", "timeseries_plotter.py", "timeseries_interface.py"],
    "utils": ["watchdog.py", "ctx_snapshot.py", "coreflow_konsolidator.py", "coreflow_manager.py"],
    "env": [".env", ".env.enc", "decrypt_env.py"],
    "docs": [".md", "README", "CONTEXT"],
    "backup": ["duplicate_report_", "context_snapshot_", "migration_log_"],
}

registry = {}
for category, patterns in CATEGORIES.items():
    registry[category] = []
    for path in ROOT_DIR.rglob("*"):
        if path.is_file():
            for pat in patterns:
                if pat in path.name:
                    registry[category].append(str(path.relative_to(ROOT_DIR)))
                    break

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_dir = ROOT_DIR / "backup"
docs_dir = ROOT_DIR / "docs"
backup_dir.mkdir(parents=True, exist_ok=True)
docs_dir.mkdir(parents=True, exist_ok=True)

# Datei 1: JSON Registry
json_out = backup_dir / f"coreflow_registry_{timestamp}.json"
with open(json_out, "w", encoding="utf-8") as f:
    json.dump(registry, f, indent=2)

# Datei 2: Markdown Strukturübersicht
md_out = docs_dir / "COREFlow_STRUCTURE.md"
lines = [f"# CoreFlow Struktur – Stand {timestamp}"]
for cat, files in registry.items():
    lines.append(f"\n## {cat.upper()}")
    if files:
        lines.extend([f"- `{f}`" for f in files])
    else:
        lines.append("_(keine Dateien gefunden)_")
with open(md_out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"\n✅ CoreFlow Rebuilder abgeschlossen.")
print(f"📄 JSON-Registry: {json_out}")
print(f"📘 Markdown:      {md_out}")
