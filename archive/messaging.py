import redis
import zmq
import logging
from datetime import datetime

class MessageBroker:
    """Redis + ZMQ Publisher für CoreFlow"""

    def __init__(self, config: dict):
        self.redis_host = config.get("redis_host", "127.0.0.1")
        self.redis_port = config.get("redis_port", 6379)
        self.redis_password = config.get("redis_password", "")
        self.zmq_target = config.get("zmq_target", "tcp://127.0.0.1:5555")
        self.use_zmq = config.get("use_zmq", True)

        self.redis = self._connect_redis()
        self.zmq_socket = self._connect_zmq() if self.use_zmq else None

    def _connect_redis(self):
        try:
            pool = redis.ConnectionPool(
                host=self.redis_host,
                port=self.redis_port,
                password=self.redis_password,
                decode_responses=True,
                socket_timeout=5
            )
            client = redis.Redis(connection_pool=pool)
            client.ping()
            logging.info("✅ Redis-Verbindung hergestellt")
            return client
        except Exception as e:
            logging.critical(f"❌ Redis-Verbindung fehlgeschlagen: {e}")
            raise

    def _connect_zmq(self):
        try:
            context = zmq.Context()
            socket = context.socket(zmq.PUB)
            socket.connect(self.zmq_target)
            logging.info(f"📡 ZMQ verbunden: {self.zmq_target}")
            return socket
        except Exception as e:
            logging.warning(f"⚠️ ZMQ konnte nicht verbunden werden: {e}")
            return None

    def publish(self, signal: dict) -> bool:
        """Signal an Redis + optional ZMQ senden"""
        try:
            channel = f"{signal['symbol'].lower()}_signals"
            self.redis.publish(channel, str(signal))
            logging.info(f"📤 Signal → Redis [{channel}]: {signal}")
        except Exception as e:
            logging.error(f"❌ Redis Publish fehlgeschlagen: {e}")
            return False

        if self.zmq_socket:
            try:
                self.zmq_socket.send_json(signal)
                logging.info(f"📤 Signal → ZMQ: {signal}")
            except Exception as e:
                logging.warning(f"⚠️ ZMQ Publish fehlgeschlagen: {e}")

        return True
