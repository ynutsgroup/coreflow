#!/usr/bin/env python3
"""
📦 RedisAI Deployment | CoreFlow
Modular FTMO-Ready Deployment Script (Lean Version)
"""

import os
import sys
import json
import hashlib
import onnx
import redisai as rai
from datetime import datetime, timezone
from dotenv import load_dotenv

# 📁 ENV & Config
load_dotenv("/opt/coreflow/.env")

CONFIG = {
    "model_path": "/opt/coreflow/models/lstm/lstm_model.onnx",
    "model_key": "lstm:trading:model",
    "audit_log": "/opt/coreflow/logs/deployment_audit.json",
    "redis_host": os.getenv("REDIS_HOST", "localhost"),
    "redis_port": int(os.getenv("REDIS_PORT", 6379)),
    "redis_password": os.getenv("REDIS_PASSWORD"),
    "ftmo_account": os.getenv("FTMO_ACCOUNT_ID", "FTMO-00000")
}

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model missing: {path}")
    with open(path, "rb") as f:
        blob = f.read()
    onnx.checker.check_model(onnx.load_from_string(blob))
    checksum = hashlib.sha256(blob).hexdigest()
    print(f"🔐 Model checksum: {checksum}")
    return blob, checksum

def connect_redis(host, port, password):
    try:
        client = rai.Client(host=host, port=port, password=password)
        if not client.ping():
            raise ConnectionError("Redis ping failed")
        print(f"🔗 Redis connected: {host}:{port}")
        return client
    except Exception as e:
        raise ConnectionError(f"Redis error: {e}")

def deploy_model(client, key, blob):
    client.modelset(
        key, backend="onnx", device="cpu",
        data=blob, inputs=["input"], outputs=["output"]
    )
    if not client.modelget(key):
        raise RuntimeError("❌ Deployment verification failed")
    print(f"✅ Model deployed to key: {key}")

def log_deployment(status, checksum):
    log = {
        "time": datetime.now(timezone.utc).isoformat(),
        "model_key": CONFIG["model_key"],
        "checksum": checksum,
        "status": status,
        "ftmo_account": CONFIG["ftmo_account"],
        "system": os.uname().nodename
    }
    os.makedirs(os.path.dirname(CONFIG["audit_log"]), exist_ok=True)
    with open(CONFIG["audit_log"], "a") as f:
        json.dump(log, f)
        f.write("\n")

if __name__ == "__main__":
    try:
        print("\n🚀 CORE FLOW RedisAI Deployment")
        model_blob, model_checksum = load_model(CONFIG["model_path"])
        redis_client = connect_redis(CONFIG["redis_host"], CONFIG["redis_port"], CONFIG["redis_password"])
        deploy_model(redis_client, CONFIG["model_key"], model_blob)
        log_deployment("success", model_checksum)
    except Exception as e:
        log_deployment(f"failed: {str(e)}", "N/A")
        print(f"❌ Deployment error: {e}")
        sys.exit(1)
