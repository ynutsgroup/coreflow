#!/usr/bin/env python3
import json
import sys

MANIFEST_JSON = """{
  "src/": [
    "coreflow_main.py",
    "valg_engine.py",
    "valg_engine_cpu.py",
    "lstm_predictor.py",
    "signal_emitter.py",
    "redis_to_zmq_bridge.py",
    "ftmo_risk_manager.py",
    "lot_calculator.py",
    "commander_control.py",
    "watchdog.py",
    "watchdog_full_monitor.py",
    "health_check.py",
    "decrypt_env.py",
    "config.py"
  ],
  "logs/": [
    "coreflow.log",
    "watchdog.log",
    "signal_emitter.log"
  ],
  ".env/": [
    ".env",
    ".env.example"
  ],
  "notes/": [
    "todo.md",
    "stand.md"
  ]
}"""

try:
    parsed = json.loads(MANIFEST_JSON)
    print("✅ JSON ist valide", file=sys.stderr)
    with open("/opt/coreflow/structure_manifest.json", "w") as f:
        json.dump(parsed, f, indent=2)
        print("💾 Gespeichert unter: /opt/coreflow/structure_manifest.json", file=sys.stderr)
except json.JSONDecodeError as e:
    print(f"❌ Fehler im Manifest: {e}", file=sys.stderr)
    sys.exit(1)
