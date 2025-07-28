import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """Core configuration class for CoreFlow"""

    # ---------- Redis Configuration ----------
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6380))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    REDIS_CHANNEL: str = os.getenv("REDIS_CHANNEL", "trading_signals")

    # ---------- Telegram Configuration ----------
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # ---------- System Configuration ----------
    LOG_DIR: Path = Path(os.getenv("LOG_DIR", "/opt/coreflow/logs"))
    GIT_REPO_PATH: Path = Path(os.getenv("GIT_REPO_PATH", "/opt/coreflow"))
    LOG_BACKUP_COUNT: int = int(os.getenv("LOG_BACKUP_COUNT", "5"))
    MAX_RESTARTS: int = int(os.getenv("MAX_RESTARTS", "5"))
    RESTART_DELAY: int = int(os.getenv("RESTART_DELAY", "30"))

    # ---------- Trading Configuration ----------
    MAX_RISK_PERCENT: float = float(os.getenv("RISK_PER_TRADE", "0.5"))
    MAX_SPREAD_RATIO: float = float(os.getenv("MAX_SPREAD_RATIO", "3.0"))
    PRICE_CHANGE_THRESHOLD: float = float(os.getenv("PRICE_CHANGE_THRESHOLD", "0.0001"))
    TRADE_COOLDOWN: int = int(os.getenv("TRADE_COOLDOWN", "60"))

    # ---------- Security ----------
    SECURE_HASH_KEY: str = os.getenv("SECURE_HASH_KEY", "default_secure_key")

    @classmethod
    def validate(cls) -> None:
        errors = []

        # Telegram Validation
        if not cls.TELEGRAM_TOKEN:
            errors.append("TELEGRAM_TOKEN is not set")
        if not cls.TELEGRAM_CHAT_ID:
            errors.append("TELEGRAM_CHAT_ID is not set")

        # Path Validation
        required_paths = [
            (cls.LOG_DIR, "Log directory"),
            (cls.GIT_REPO_PATH, "Git repository path")
        ]
        for path, name in required_paths:
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    path.chmod(0o755)
                except Exception as e:
                    errors.append(f"{name} could not be created: {e}")

        # SECURE_HASH_KEY Soft Check
        if cls.SECURE_HASH_KEY == "default_secure_key":
            print("⚠️  WARNUNG: SECURE_HASH_KEY steht noch auf 'default_secure_key' – bitte in .env.secure ändern.")

        if errors:
            error_msg = "Configuration warnings:\n" + "\n".join(f"• {error}" for error in errors)
            print(f"❌ Konfigurationswarnungen:\n{error_msg}")

    @classmethod
    def get_all_settings(cls) -> Dict[str, Any]:
        return {
            'redis': {
                'host': cls.REDIS_HOST,
                'port': cls.REDIS_PORT,
                'channel': cls.REDIS_CHANNEL,
                'password': '*****' if cls.REDIS_PASSWORD else ''
            },
            'telegram': {
                'token': '*****' if cls.TELEGRAM_TOKEN else '',
                'chat_id': cls.TELEGRAM_CHAT_ID
            },
            'system': {
                'log_dir': str(cls.LOG_DIR),
                'git_repo': str(cls.GIT_REPO_PATH),
                'log_backups': cls.LOG_BACKUP_COUNT,
                'max_restarts': cls.MAX_RESTARTS,
                'restart_delay': cls.RESTART_DELAY
            },
            'trading': {
                'max_risk': cls.MAX_RISK_PERCENT,
                'spread_ratio': cls.MAX_SPREAD_RATIO,
                'price_change_threshold': cls.PRICE_CHANGE_THRESHOLD,
                'cooldown': cls.TRADE_COOLDOWN
            }
        }

try:
    Config.validate()
except Exception as e:
    print(f"[WARN] Config Check: {e}")
