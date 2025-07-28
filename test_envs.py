#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoreFlow .env.enc Checker
✅ Findet und testet alle .env.enc-Dateien im System
✅ Unterstützt symbolische Links & modulares Vault-Design
"""

import sys
from pathlib import Path

# === Dynamischer Pfad für decrypt_env.py ===
decrypt_path = Path("/opt/coreflow/utils")
if decrypt_path.exists():
    sys.path.insert(0, str(decrypt_path))
else:
    raise FileNotFoundError(f"❌ Pfad nicht gefunden: {decrypt_path}")

from decrypt_env import test_all_envs  # 🔐 Importiere zentrale Prüf-Funktion

if __name__ == "__main__":
    test_all_envs("/opt/coreflow")
