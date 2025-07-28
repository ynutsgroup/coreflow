#!/usr/bin/env python3
import sqlite3
from datetime import datetime

DB_PATH = "/opt/coreflow/status/data/coreflow_modules.db"
HTML_PATH = "/opt/coreflow/status/outputs/status.html"

def generate_dashboard():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Daten abfragen
    cursor.execute("SELECT COUNT(*) FROM modules")
    total_modules = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM modules WHERE status != 'unreviewed'")
    reviewed_modules = cursor.fetchone()[0]
    
    cursor.execute("SELECT path, status FROM modules WHERE status != 'unreviewed' ORDER BY last_updated DESC LIMIT 5")
    recent_updates = cursor.fetchall()
    
    # HTML generieren
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CoreFlow Modul Status</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ display: flex; align-items: center; margin-bottom: 30px; }}
        .logo {{ height: 50px; margin-right: 20px; }}
        .progress-bar {{ 
            width: 100%; 
            background-color: #f0f0f0;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .progress {{ 
            width: {int(reviewed_modules/total_modules*100)}%; 
            background-color: #4CAF50;
            height: 30px;
            border-radius: 5px;
            text-align: center;
            line-height: 30px;
            color: white;
        }}
        .module-list {{ margin-top: 20px; }}
        .module {{ 
            padding: 10px; 
            margin: 5px 0; 
            background-color: #f9f9f9;
            border-left: 4px solid #4CAF50;
        }}
    </style>
</head>
<body>
    <div class="header">
        <img src="logos/coreflow_logo.png" class="logo" alt="CoreFlow Logo">
        <h1>CoreFlow Modul Status</h1>
    </div>
    
    <div class="progress-bar">
        <div class="progress">{reviewed_modules}/{total_modules} Module reviewed</div>
    </div>
    
    <h2>Recently Updated Modules</h2>
    <div class="module-list">
        {"".join(f'<div class="module"><strong>{path}</strong> - Status: {status}</div>' for path, status in recent_updates)}
    </div>
    
    <p>Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
</body>
</html>
    """
    
    with open(HTML_PATH, 'w') as f:
        f.write(html_content)
    
    print(f"Dashboard generated at {HTML_PATH}")

if __name__ == "__main__":
    generate_dashboard()
