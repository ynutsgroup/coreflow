#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, socket, argparse, time, hashlib
from datetime import datetime
from pathlib import Path

HOST = socket.gethostname()
NOW = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
BACKUP_DIR = Path("/opt/coreflow/backup")
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# Zentrale “Single Source of Truth” für Kernpfade (Linux)
PATHS = {
    # Core
    "coreflow_main.py": "/opt/coreflow/core/coreflow_main.py",
    "valg_engine.py": "/opt/coreflow/core/valg_engine.py",
    "lstm_predictor.py": "/opt/coreflow/core/lstm_predictor.py",
    "envloader.py": "/opt/coreflow/utils/envloader.py",
    "decrypt_env.py": "/opt/coreflow/utils/decrypt_env.py",

    # Timeseries
    "timeseries_interface.py": "/opt/coreflow/modules/timeseries/timeseries_interface.py",
    "ts_aggregator.py": "/opt/coreflow/modules/timeseries/ts_aggregator.py",
    "timeseries_plotter.py": "/opt/coreflow/modules/timeseries/timeseries_plotter.py",
    "market_hours_guard.py": "/opt/coreflow/utils/market_hours_guard.py",

    # Signaling & Exec
    "signal_emitter.py": "/opt/coreflow/core/signal_emitter.py",
    "signal_receiver_bridge.md": "/opt/coreflow/docs/signal_receiver_bridge.md",

    # Watchdog / Ops
    "watchdog.py": "/opt/coreflow/utils/watchdog.py",
    ".env.enc": "/opt/coreflow/.env.enc",
    ".env": "/opt/coreflow/.env",

    # Docs
    "PROJECT_README.md": "/opt/coreflow/PROJECT_README.md",
    "COREFlow_CONTEXT.md": "/opt/coreflow/COREFlow_CONTEXT.md",
}

def file_info(p: str):
    pth = Path(p)
    exists = pth.exists()
    info = {
        "path": p,
        "exists": exists,
        "size": None,
        "mtime": None,
        "sha1": None
    }
    if exists and pth.is_file():
        try:
            info["size"] = pth.stat().st_size
            info["mtime"] = datetime.utcfromtimestamp(pth.stat().st_mtime).strftime("%Y-%m-%dT%H:%M:%SZ")
            # Nur kleine Dateien hashen (bis 2 MB)
            if info["size"] is not None and info["size"] <= 2_000_000:
                h = hashlib.sha1()
                with open(p, "rb") as f:
                    h.update(f.read())
                info["sha1"] = h.hexdigest()
        except Exception as e:
            info["error"] = f"stat/hash failed: {e}"
    return info

def build_snapshot(extra_pairs):
    paths = dict(PATHS)
    for name, path in extra_pairs:
        paths[name] = path

    details = {name: file_info(path) for name, path in paths.items()}

    summary_ok = [n for n, v in details.items() if v["exists"]]
    summary_missing = [n for n, v in details.items() if not v["exists"]]

    snap = {
        "meta": {
            "host": HOST,
            "created_utc": NOW,
            "python": sys.version.split()[0],
        },
        "paths": details,
        "summary": {
            "ok": summary_ok,
            "missing": summary_missing,
            "counts": {"ok": len(summary_ok), "missing": len(summary_missing), "total": len(details)},
        },
    }
    return snap

def write_files(snap, quick=False):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_json = BACKUP_DIR / f"context_snapshot_{ts}.json"
    latest = BACKUP_DIR / "context_snapshot_latest.json"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2)

    # “latest” aktualisieren (atomar)
    tmp = BACKUP_DIR / f".context_snapshot_tmp_{ts}.json"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2)
    tmp.replace(latest)

    # Ausgabe
    if quick:
        ok = snap["summary"]["ok"]
        missing = snap["summary"]["missing"]
        print(f"[CTX {snap['meta']['created_utc']}] OK({len(ok)}): {', '.join(ok)}")
        if missing:
            print(f"[CTX {snap['meta']['created_utc']}] MISSING({len(missing)}): {', '.join(missing)}")
        print(f"Saved: {out_json}  |  Latest: {latest}")
    else:
        print(json.dumps(snap, ensure_ascii=False, indent=2))
        print(f"\nSaved: {out_json}\nLatest: {latest}")

def main():
    ap = argparse.ArgumentParser(description="COREFLOW Kontext-Snapshot")
    ap.add_argument("--quick", action="store_true", help="Nur Kurz-Output + Dateispeicherung")
    ap.add_argument("--add", action="append", default=[], metavar="NAME=PATH",
                    help="Zusätzlichen Eintrag mappen (mehrfach möglich)")
    args = ap.parse_args()

    extras = []
    for pair in args.add:
        if "=" in pair:
            name, path = pair.split("=", 1)
            extras.append((name.strip(), path.strip()))
        else:
            print(f"Warn: --add erwartet NAME=PATH, erhalten: {pair}", file=sys.stderr)

    snap = build_snapshot(extras)
    write_files(snap, quick=args.quick)

if __name__ == "__main__":
    main()
