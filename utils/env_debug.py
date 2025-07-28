#!/usr/bin/env python3
import os
from utils.decrypt_env import load_encrypted_env

if not load_encrypted_env():
    print("❌ .env.enc konnte nicht geladen werden")
    exit(1)

for key in sorted(os.environ.keys()):
    if "PASS" in key or "KEY" in key:
        print(f"{key}=********")
    else:
        print(f"{key}={os.environ[key]}")
