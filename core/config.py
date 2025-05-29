import os
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from /opt/coreflow/.env
load_dotenv(dotenv_path="/opt/coreflow/.env")

class Config:
    """Core configuration class for CoreFlow"""

    # ---------- Redis Configuration ----------
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    REDIS_CHANNEL: str = os.getenv("REDIS_CHANNEL", "trading_signals")

    # ---------- Telegram Configuration ----------
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # ---------- System Configuration ----------
    LOG_DIR: Path = Path(os.getenv("LOG_DIR", "/opt/coreflow/logs"))
    GIT_REPO_PATH: Path = Path(os.getenv("GIT_REPO_PATH", "/opt/coreflow"))
    LOG_BACKUP_COUNT: int = int(os.getenv("LOG_BACKUP_COUNT", "7"))
    MAX_RESTARTS: int = int(os.getenv("MAX_RESTARTS", "5"))
    RESTART_DELAY: int = int(os.getenv("RESTART_DELAY", "30"))

    # ---------- Trading Configuration ----------
    MAX_RISK_PERCENT: float = float(os.getenv("RISK_PER_TRADE", "0.5"))
    MAX_SPREAD_RATIO: float = float(os.getenv("MAX_SPREAD_RATIO", "3.0"))
    PRICE_CHANGE_THRESHOLD: float = float(os.getenv("PRICE_CHANGE_THRESHOLD", "0.0001"))
    TRADE_COOLDOWN: int = int(os.getenv("TRADE_COOLDOWN", "60"))

    # ---------- Security Settings ----------
    SECURE_HASH_KEY: str = os.getenv("SECURE_HASH_KEY", "default_secure_key")

    @classmethod
    def validate(cls) -> None:
        """Validate critical configuration values"""
        errors = []

        # Telegram validation
        if not cls.TELEGRAM_TOKEN:
            errors.append("TELEGRAM_TOKEN is not set")
        if not cls.TELEGRAM_CHAT_ID:
            errors.append("TELEGRAM_CHAT_ID is not set")

        # Path validation
        required_paths = [
            (cls.LOG_DIR, "Log directory"),
            (cls.GIT_REPO_PATH, "Git repository path")
        ]

        for path, name in required_paths:
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    cls.LOG_DIR.chmod(0o755)
                except Exception as e:
                    errors.append(f"{name} does not exist and could not be created: {str(e)}")

        # Security validation
        if cls.SECURE_HASH_KEY == "default_secure_key":
            errors.append("SECURE_HASH_KEY should be changed from default value")

        if errors:
            error_msg = "Configuration errors:\n" + "\n".join(f"• {error}" for error in errors)
            raise ValueError(error_msg)

    @classmethod
    def get_all_settings(cls) -> Dict[str, Any]:
        """Return all configuration settings as a dictionary"""
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

    def __str__(self) -> str:
        """Human-readable configuration"""
        from pprint import pformat
        return pformat(self.get_all_settings())

# Validate configuration when imported
try:
    Config.validate()
except ValueError as e:
    print(f"❌ Configuration error: {e}")
    raise

# Example usage:
if __name__ == "__main__":
    print("Current configuration:")
    print(Config.get_all_settings())

# New Feature: Telegram confirmation after /restart
if __name__ == "__main__":
    import requests
    def send_restart_confirmation():
        try:
            requests.post(
                f"https://api.telegram.org/bot{Config.TELEGRAM_TOKEN}/sendMessage",
                json={
                    "chat_id": Config.TELEGRAM_CHAT_ID,
                    "text": "♻️ CoreFlow wurde erfolgreich neu gestartet!",
                    "parse_mode": "HTML"
                },
                timeout=5
            )
        except Exception as e:
            print(f"❌ Failed to send Telegram confirmation: {e}")

    send_restart_confirmation()
