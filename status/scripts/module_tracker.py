#!/usr/bin/env python3
import sqlite3
import os
from datetime import datetime

DB_PATH = "/opt/coreflow/status/data/coreflow_modules.db"

def scan_modules():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS modules (
        path TEXT PRIMARY KEY,
        status TEXT DEFAULT 'unreviewed',
        last_updated TEXT,
        author TEXT DEFAULT 'unknown'
    )""")
    
    for root, _, files in os.walk("/opt/coreflow"):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                cursor.execute("""INSERT OR REPLACE INTO modules 
                                (path, last_updated) VALUES (?, ?)""", 
                                (path, datetime.now().isoformat()))
    conn.commit()

if __name__ == "__main__":
    scan_modules()
    print("[OK] Module gescannt! DB aktualisiert.")
