#!/usr/bin/env python3
import os
import time

def find_latest_env_enc(root_dir="/"):
    latest_path = None
    latest_mtime = 0

    for root, dirs, files in os.walk(root_dir):
        if ".env.enc" in files:
            full_path = os.path.join(root, ".env.enc")
            mtime = os.path.getmtime(full_path)
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_path = full_path

    if latest_path:
        readable_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(latest_mtime))
        print(f"✅ Neueste .env.enc gefunden:\n{latest_path} (Letzte Änderung: {readable_time})")
    else:
        print("❌ Keine .env.enc-Datei gefunden.")

if __name__ == "__main__":
    find_latest_env_enc("/opt/coreflow/")
