#!/usr/bin/env python3
"""
CoreFlow Institutional Snapshot Agent (v3.2)
FTMO-Compliant System Diagnostics & Infrastructure Monitoring
"""

import os
import sys
import json
import socket
import logging
import datetime
import subprocess
from pathlib import Path
from typing import Dict, Any

import psutil
import GPUtil
import redis
from dotenv import load_dotenv

# === CONFIG ===
class CoreConfig:
    CORE_PATH = Path("/opt/coreflow")
    ENV_FILE = CORE_PATH / ".env"
    BACKUP_DIR = CORE_PATH / "backup"
    LOG_DIR = CORE_PATH / "logs"
    SNAPSHOT_JSON = BACKUP_DIR / "cfm_snapshot.json"
    SNAPSHOT_MD = BACKUP_DIR / "cfm_snapshot.md"
    LOG_FILE = LOG_DIR / "cfm_assistant.log"

    REQUIRED_ENV = [
        "SYMBOL",
        "MODE",
        "REDIS_HOST",
        "REPLAY_FILE"
    ]

# === LOGGING ===
class SecureLogger:
    def __init__(self):
        self.logger = logging.getLogger("CFM-Institutional")
        self._setup_logging()

    def _setup_logging(self):
        CoreConfig.LOG_DIR.mkdir(exist_ok=True, parents=True)
        CoreConfig.BACKUP_DIR.mkdir(exist_ok=True, parents=True)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(module)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S%z"
        )

        handlers = [
            logging.FileHandler(CoreConfig.LOG_FILE, encoding='utf-8'),
            logging.StreamHandler()
        ]

        for handler in handlers:
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.setLevel(logging.INFO)

# === ENV ===
class EnvManager:
    @staticmethod
    def load_env() -> bool:
        """Load and validate environment"""
        if not CoreConfig.ENV_FILE.exists():
            SecureLogger().logger.critical(f"❌ ENV file missing: {CoreConfig.ENV_FILE}")
            return False

        try:
            load_dotenv(CoreConfig.ENV_FILE)
            missing = [var for var in CoreConfig.REQUIRED_ENV if not os.getenv(var)]
            if missing:
                SecureLogger().logger.error(f"Missing required ENV vars: {missing}")
                return False
            SecureLogger().logger.info(f"✅ ENV loaded: {CoreConfig.ENV_FILE}")
            return True
        except Exception as e:
            SecureLogger().logger.critical(f"ENV load failed: {str(e)}")
            return False

    @staticmethod
    def getenv(key: str, default=None, cast=str) -> Any:
        val = os.getenv(key, default)
        try:
            return cast(val)
        except (ValueError, TypeError):
            return val

# === SYSTEM CHECKS ===
class SystemProbe:
    @staticmethod
    def check_redis() -> Dict[str, Any]:
        try:
            conn = redis.Redis(
                host=EnvManager.getenv("REDIS_HOST"),
                port=EnvManager.getenv("REDIS_PORT", 6379, int),
                password=EnvManager.getenv("REDIS_PASSWORD"),
                socket_timeout=3,
                health_check_interval=30
            )
            latency = conn.ping()
            info = conn.info()
            return {
                "status": "OK",
                "latency_ms": round(latency * 1000, 2),
                "version": info.get('redis_version'),
                "used_memory": info.get('used_memory_human')
            }
        except Exception as e:
            return {"status": f"ERROR: {str(e)}"}

    @staticmethod
    def gpu_status() -> Dict[str, Any]:
        try:
            gpus = GPUtil.getGPUs()
            return {
                "count": len(gpus),
                "gpus": [
                    {
                        "id": gpu.id,
                        "name": gpu.name,
                        "load": gpu.load,
                        "memory": f"{gpu.memoryUsed}/{gpu.memoryTotal}MB"
                    } for gpu in gpus
                ]
            }
        except:
            return {"count": 0}

    @staticmethod
    def check_ai_assets() -> Dict[str, str]:
        return {
            "valg_engine": "found" if Path(EnvManager.getenv("VALG_ENGINE_PATH", f"{CoreConfig.CORE_PATH}/core/valg_engine.py")).exists() else "missing",
            "lstm_model": "present" if Path(EnvManager.getenv("LSTM_MODEL_PATH", f"{CoreConfig.CORE_PATH}/models/lstm_model.h5")).exists() else "missing",
            "tensorrt": "available" if SystemProbe._check_tensorrt() else "unavailable"
        }

    @staticmethod
    def _check_tensorrt() -> bool:
        try:
            import tensorrt
            return True
        except ImportError:
            return False

# === SNAPSHOT ===
class SnapshotEngine:
    @staticmethod
    def generate() -> None:
        snapshot = {
            "metadata": {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "hostname": socket.gethostname(),
                "coreflow_version": SnapshotEngine._get_coreflow_version()
            },
            "environment": {
                "symbol": EnvManager.getenv("SYMBOL"),
                "mode": EnvManager.getenv("MODE"),
                "replay_file": EnvManager.getenv("REPLAY_FILE"),
                "env_file": str(CoreConfig.ENV_FILE)
            },
            "infrastructure": {
                "redis": SystemProbe.check_redis(),
                "zmq": {
                    "enabled": EnvManager.getenv("ZMQ_ENABLED", "false").lower() == "true",
                    "port": EnvManager.getenv("ZMQ_PORT", 5555, int)
                },
                "gpu": SystemProbe.gpu_status(),
                "system_load": {
                    "cpu": psutil.cpu_percent(),
                    "memory": psutil.virtual_memory()._asdict()
                }
            },
            "ai_components": SystemProbe.check_ai_assets(),
            "paths": {
                "log_file": str(CoreConfig.LOG_FILE),
                "backup_dir": str(CoreConfig.BACKUP_DIR)
            }
        }

        with open(CoreConfig.SNAPSHOT_JSON, 'w') as f:
            json.dump(snapshot, f, indent=2)

        SnapshotEngine._create_markdown_report(snapshot)

    @staticmethod
    def _create_markdown_report(data: Dict[str, Any]) -> None:
        md_content = f"""# CoreFlow Institutional Snapshot

## System Overview
- **Timestamp**: `{data['metadata']['timestamp']}`
- **Host**: `{data['metadata']['hostname']}`
- **CoreFlow Version**: `{data['metadata']['coreflow_version']}`

## Trading Environment
- **Symbol**: `{data['environment']['symbol']}`
- **Mode**: `{data['environment']['mode']}`
- **Replay File**: `{data['environment']['replay_file']}`

## Infrastructure Status
### Redis
- **Host**: `{EnvManager.getenv('REDIS_HOST')}`
- **Port**: `{EnvManager.getenv('REDIS_PORT')}`
- **Status**: `{data['infrastructure']['redis']['status']}`
- **Version**: `{data['infrastructure']['redis'].get('version', 'N/A')}`
- **Memory**: `{data['infrastructure']['redis'].get('used_memory', 'N/A')}`

### GPU Resources
- **Count**: `{data['infrastructure']['gpu']['count']}`
{SnapshotEngine._format_gpu_info(data['infrastructure']['gpu'])}

## AI Components
- **VALG Engine**: `{data['ai_components']['valg_engine']}`
- **LSTM Model**: `{data['ai_components']['lstm_model']}`
- **TensorRT**: `{data['ai_components']['tensorrt']}`
"""
        with open(CoreConfig.SNAPSHOT_MD, 'w') as f:
            f.write(md_content)

    @staticmethod
    def _format_gpu_info(gpu_data: Dict) -> str:
        if gpu_data['count'] == 0:
            return "- **GPUs**: None detected"
        return "\n".join(
            f"- **GPU {gpu['id']}**: {gpu['name']} (Load: {gpu['load']*100:.1f}%, Memory: {gpu['memory']})"
            for gpu in gpu_data['gpus']
        )

    @staticmethod
    def _get_coreflow_version() -> str:
        try:
            result = subprocess.run(
                ["coreflow", "--version"],
                capture_output=True,
                text=True
            )
            return result.stdout.strip() or "unknown"
        except:
            return "unknown"

# === EXECUTION ===
if __name__ == "__main__":
    logger = SecureLogger().logger

    if not EnvManager.load_env():
        sys.exit(1)

    try:
        SnapshotEngine.generate()
        logger.info("✅ Institutional snapshot completed")
    except Exception as e:
        logger.critical(f"❌ Snapshot failed: {str(e)}", exc_info=True)
        sys.exit(1)
