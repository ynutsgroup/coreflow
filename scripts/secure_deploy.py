"""
CORE FLOW RedisAI Model Deployment
FTMO-Compliant Implementation with Secure Authentication
"""

import redisai as rai
import hashlib
import onnx
from datetime import datetime, timezone
import json
import os
from dotenv import load_dotenv

# 🔐 Load secure environment variables
load_dotenv('/opt/coreflow/.env')  # Load from your encrypted env

# 🔧 Configuration from .env
CONFIG = {
    "redis_host": os.getenv("REDIS_HOST", "localhost"),
    "redis_port": int(os.getenv("REDIS_PORT", "6379")),
    "redis_password": os.getenv("REDIS_PASSWORD"),
    "model_key": "lstm:trading:model",
    "model_path": "/opt/coreflow/models/lstm/lstm_model.onnx",
    "audit_log": "/opt/coreflow/logs/deployment_audit.json",
    "socket_timeout": int(os.getenv("REDIS_SOCKET_TIMEOUT", "30")),
    "ftmo_account": os.getenv("FTMO_ACCOUNT_ID")
}

class SecureModelDeployer:
    """FTMO-compliant model deployment with audit trail"""
    
    def __init__(self):
        self.connection = None
        self.checksum = None
        self.model_blob = None
        
    def _verify_environment(self):
        """Validate all required environment variables"""
        required_vars = ['REDIS_HOST', 'REDIS_PASSWORD', 'FTMO_ACCOUNT_ID']
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise EnvironmentError(f"Missing required env vars: {', '.join(missing)}")

    def _load_model(self):
        """Secure model loading with checksum verification"""
        if not os.path.exists(CONFIG["model_path"]):
            raise FileNotFoundError(f"Model file not found at {CONFIG['model_path']}")
            
        with open(CONFIG["model_path"], "rb") as f:
            self.model_blob = f.read()
        
        # Validate ONNX model
        onnx_model = onnx.load_from_string(self.model_blob)
        onnx.checker.check_model(onnx_model)
        
        self.checksum = hashlib.sha256(self.model_blob).hexdigest()
        print(f"🔐 Model Checksum: {self.checksum}")
        
    def _connect_redis(self):
        """Authenticated connection with FTMO compliance checks"""
        try:
            self.connection = rai.Client(
                host=CONFIG["redis_host"],
                port=CONFIG["redis_port"],
                password=CONFIG["redis_password"],
                socket_timeout=CONFIG["socket_timeout"]
            )
            
            # Verify connection matches FTMO account
            ftmo_key = f"ftmo:{CONFIG['ftmo_account']}:status"
            if not self.connection.exists(ftmo_key):
                raise ConnectionError("FTMO account validation failed")
                
            print("🔑 Successfully authenticated with RedisAI")
            
        except Exception as e:
            raise ConnectionError(f"Redis connection failed: {str(e)}")
    
    def _log_deployment(self, status):
        """FTMO-compliant audit logging"""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ftmo_account": CONFIG["ftmo_account"],
            "model": {
                "key": CONFIG["model_key"],
                "checksum": self.checksum,
                "path": CONFIG["model_path"]
            },
            "risk_parameters": {
                "max_risk": os.getenv("MAX_RISK_PERCENT"),
                "daily_loss_limit": os.getenv("DAILY_LOSS_LIMIT_PERCENT")
            },
            "status": status
        }
        
        try:
            os.makedirs(os.path.dirname(CONFIG["audit_log"]), exist_ok=True)
            mode = 'a' if os.path.exists(CONFIG["audit_log"]) else 'w'
            with open(CONFIG["audit_log"], mode) as f:
                json.dump(log_entry, f, indent=2)
                f.write('\n')  # Newline for log rotation
        except Exception as e:
            print(f"⚠️ Audit logging error: {str(e)}")

    def deploy(self):
        """Secure deployment workflow"""
        try:
            print("🚀 Initializing FTMO-compliant model deployment...")
            
            # 1. Environment verification
            self._verify_environment()
            
            # 2. Load and validate model
            self._load_model()
            
            # 3. Establish secure connection
            self._connect_redis()
            
            # 4. Deploy model
            self.connection.modelset(
                CONFIG["model_key"],
                backend="onnx",
                device="cpu",
                inputs=["input"],
                outputs=["output"],
                data=self.model_blob
            )
            
            # 5. Post-deployment verification
            if not self.connection.modelget(CONFIG["model_key"]):
                raise RuntimeError("Model verification failed")
            
            # 6. Finalize audit log
            self._log_deployment("success")
            print(f"✅ Model successfully deployed to {CONFIG['redis_host']} | FTMO Account: {CONFIG['ftmo_account']}")
            
            return True
            
        except Exception as e:
            self._log_deployment(f"failed: {str(e)}")
            print(f"❌ Deployment failed: {str(e)}")
            return False

# 🛡️ Secure execution
if __name__ == "__main__":
    deployer = SecureModelDeployer()
    if not deployer.deploy():
        exit(1)
