import os
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Union

log = logging.getLogger(__name__)

class EnvLoader:
    """Secure and type-aware environment loader for CoreFlow + FTMO compliance."""

    def __init__(self, env_path: Union[str, Path] = "/opt/coreflow/.env"):
        self.env_path = Path(env_path)
        self.env: Dict[str, Union[str, int, float, bool]] = {}
        self._validation_patterns = {
            'API_KEY': r'^[a-f0-9]{64}$',
            'TRADING_ACCOUNT': r'^FTMO-\d{5}$',
            'DB_URL': r'^postgresql://[^:]+:[^@]+@[^/]+/\w+$'
        }

    def load(self) -> None:
        if not self.env_path.exists():
            log.warning(f"[EnvLoader] .env not found: {self.env_path}")
            return

        try:
            with self.env_path.open("r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    if any(token in line for token in [";", "`", "$(", "\\", "[", "]"]):
                        log.warning(f"[EnvLoader] Ignoring unsafe line {line_number}")
                        continue

                    if "=" not in line:
                        log.warning(f"[EnvLoader] Invalid format at line {line_number}: {line}")
                        continue

                    key, val = line.split("=", 1)
                    key, val = key.strip(), val.strip().strip('"').strip("'")

                    if key in self._validation_patterns:
                        if not re.match(self._validation_patterns[key], val):
                            log.error(f"[EnvLoader] Invalid format for {key} at line {line_number}")
                            continue

                    self._set_env(key, val)

        except Exception as e:
            log.exception(f"[EnvLoader] Error loading .env: {e}")
            raise

    def _set_env(self, key: str, value: str) -> None:
        # Boolean
        if value.lower() in ("true", "false"):
            parsed = value.lower() == "true"
        # Integer
        elif value.isdigit():
            parsed = int(value)
        # Float
        elif re.fullmatch(r"\d+\.\d+", value):
            parsed = float(value)
        else:
            parsed = value

        self.env[key] = parsed
        os.environ[key] = str(parsed)

    def get(self, key: str, default: Optional[Union[str, int, float, bool]] = None) -> Union[str, int, float, bool, None]:
        val = self.env.get(key, os.getenv(key, default))
        if isinstance(default, bool):
            return str(val).lower() in ("true", "1", "yes", "y")
        if isinstance(default, int) and str(val).isdigit():
            return int(val)
        if isinstance(default, float) and re.fullmatch(r"\d+\.\d+", str(val)):
            return float(val)
        return val

    def validate_ftmo_requirements(self) -> bool:
        required = ['FTMO_API_KEY', 'FTMO_ACCOUNT_ID', 'RISK_PERCENTAGE']
        missing = [k for k in required if not self.get(k)]
        if missing:
            log.error(f"[EnvLoader] Missing FTMO keys: {missing}")
            return False

        try:
            risk = float(self.get("RISK_PERCENTAGE"))
            if not 0.01 <= risk <= 5.0:
                raise ValueError
        except Exception:
            log.error("[EnvLoader] RISK_PERCENTAGE must be between 0.01 and 5.0")
            return False

        return True

    def get_ai_config(self) -> Dict[str, Union[str, int, float, bool]]:
        return {
            "model_path": self.get("AI_MODEL_PATH", "/opt/models/default"),
            "inference_batch": self.get("AI_BATCH_SIZE", 32),
            "confidence_threshold": self.get("AI_CONFIDENCE", 0.95),
            "enable_ml": self.get("AI_ENABLED", False)
        }
