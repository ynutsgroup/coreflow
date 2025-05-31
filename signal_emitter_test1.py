from dotenv import load_dotenv
import os
import redis
import json
import logging
from cryptography.fernet import Fernet

# Lade Umgebungsvariablen aus der .env-Datei
load_dotenv()

class SignalEmitter:
    def __init__(self):
        self.setup_logging()
        self.signal_dir = Path('/opt/coreflow/signals')
        self.signal_dir.mkdir(exist_ok=True)
        self.signal_count = 0
        self.setup_encryption()
        self.setup_redis()

    def setup_logging(self):
        """Configure logging"""
        log_dir = Path('/opt/coreflow/logs')
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir/'signal_emitter.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('SignalEmitter')

    def setup_redis(self):
        """Initialize Redis with error handling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                redis_password = os.getenv('REDIS_PASSWORD', 'fallback_password')  # Laden des Passworts aus der .env-Datei
                
                self.redis = redis.Redis(
                    host="127.0.0.1",
                    port=6379,
                    password=redis_password,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                    health_check_interval=30
                )
                
                # Test connection with safe command
                self.redis.client_id()
                self.logger.info("Redis connection established")
                return
                
            except redis.exceptions.ResponseError as e:
                if "MISCONF" in str(e):
                    self.logger.warning("Redis configuration issue detected")
                    try:
                        self.redis.config_set('stop-writes-on-bgsave-error', 'no')
                        self.logger.warning("Temporarily disabled write protection")
                        continue
                    except Exception as config_error:
                        self.logger.error(f"Failed to disable write protection in Redis: {str(config_error)}")
                self.logger.error(f"Redis error (attempt {attempt+1}/{max_retries}): {str(e)}")
                time.sleep(2)
            except Exception as e:
                self.logger.error(f"Connection error (attempt {attempt+1}/{max_retries}): {str(e)}")
                time.sleep(2)
        
        self.logger.error("Failed to establish Redis connection after retries")
        raise ConnectionError("Could not connect to Redis")

    def setup_encryption(self):
        """Initialize encryption with key rotation"""
        key_path = Path("/opt/coreflow/.env.key")
        if not key_path.exists():
            self.logger.info("Generating new encryption key")
            key = Fernet.generate_key()
            with open(key_path, "wb") as f:
                f.write(key)
            key_path.chmod(0o600)
        
        with open(key_path, "rb") as f:
            key = f.read()
        self.fernet = Fernet(key)
        self.logger.info("Encryption initialized")

    def create_signal(self, signal_type, data):
        """Generate and publish signal with fallback"""
        signal_id = f"{int(time.time())}_{self.signal_count}"
        signal_data = {
            'id': signal_id,
            'type': signal_type,
            'timestamp': time.time(),
            'data': data
        }
        
        # Save to file (local fallback)
        signal_file = self.signal_dir / f"{signal_id}.json"
        try:
            with open(signal_file, 'w') as f:
                json.dump(signal_data, f)
        except Exception as e:
            self.logger.error(f"Failed to save signal {signal_id}: {str(e)}")
            return None

        # Publish to Redis
        try:
            payload = json.dumps(signal_data).encode("utf-8")
            encrypted_payload = self.fernet.encrypt(payload)
            self.redis.publish("trading_signals", encrypted_payload)
            self.logger.info(f"Published signal {signal_id}")
            return signal_file
        except Exception as e:
            self.logger.error(f"Failed to publish signal {signal_id}: {str(e)}")
            return signal_file  # Return file even if Redis failed

    def run(self):
        """Main loop with graceful degradation"""
        self.logger.info("Signal Emitter started")
        while True:
            try:
                signal_file = self.create_signal(
                    signal_type='heartbeat',
                    data={
                        'status': 'ok',
                        'load': os.getloadavg(),
                        'memory': os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
                    }
                )
                
                if signal_file:
                    self.signal_count += 1
                else:
                    self.logger.warning("Signal creation failed, retrying...")
                
                time.sleep(10)
                
            except KeyboardInterrupt:
                self.logger.info("Shutting down...")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error: {str(e)}")
                time.sleep(5)  # Prevent tight failure loops

if __name__ == "__main__":
    emitter = SignalEmitter()
    emitter.run()
