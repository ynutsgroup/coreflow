#!/usr/bin/env python3
"""
Refactoring Script – Sanitize CoreFlow Scripts
Targets:
- Remove hardcoded IPs, passwords, ports, or model keys
- Replace with dynamic .env reads
- Enforce professional-grade institutional configuration
"""

import re
import os
from pathlib import Path

# Define the root directory to process
ROOT_DIR = Path("/opt/coreflow/")

# Secure keys from ENV
SECURE_KEYS = {
    "REDIS_HOST": "os.getenv('REDIS_HOST')",
    "REDIS_PORT": "int(os.getenv('REDIS_PORT', 6379))",
    "REDIS_PASSWORD": "os.getenv('REDIS_PASSWORD')",
    "MODEL_NAME": "os.getenv('MODEL_NAME') or 'lstm:trading:model'",
    "REDIS_PASS": "os.getenv('REDIS_PASS')",
    "ZMQ_BIND_ADDR": "os.getenv('ZMQ_BIND_ADDR', 'tcp://*:5558')"
}

# Patterns to detect hardcoded strings
hardcoded_ip = re.compile(r"(?<![\w])(?:\d{1,3}\.){3}\d{1,3}(?![\w])")
hardcoded_port = re.compile(r"(?<!\w)(6379|6380|5556|5558)(?!\w)")
hardcoded_password = re.compile(r"(['\"])(ncasInitHW!|s3cur3P@ss!|J\$v91x!Jq1Gz)(['\"])")
hardcoded_model = re.compile(r"(['\"])(lstm:trading:model)(['\"])" )

# Patch function
def patch_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    patched = False
    new_lines = []
    for line in lines:
        original = line
        line = hardcoded_ip.sub("os.getenv('REDIS_HOST')", line)
        line = hardcoded_port.sub("int(os.getenv('REDIS_PORT'))", line)
        line = hardcoded_password.sub("os.getenv('REDIS_PASSWORD')", line)
        line = hardcoded_model.sub("os.getenv('MODEL_NAME')", line)

        if line != original:
            patched = True
        new_lines.append(line)

    if patched:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"✅ Patched: {file_path}")
    else:
        print(f"⏭️ No changes: {file_path}")

# Walk through files
for root, _, files in os.walk(ROOT_DIR):
    for file in files:
        if file.endswith('.py') or file.endswith('.sh'):
            patch_file(Path(root) / file)
