#!/usr/bin/env python3
"""
CORE FLOW BRIDGE SERVICE - FINAL FIXED VERSION
"""

import os
import time
import logging
import redis
import zmq
from pathlib import Path
from cryptography.fernet import Fernet
from dotenv import load_dotenv

class SecureConfig:
    def __init__(self):
        self.env_path = Path('/opt/coreflow/.env')
        load_dotenv(self.env_path, override=True)

class CoreBridge:
    def __init__(self):
        self.config = SecureConfig()
        self.logger = self._setup_logging()
        self.redis = self._init_redis()
        self.zmq_ctx, self.zmq_socket = self._init_zmq()
        self.logger.info("Service initialisiert")

    def _setup_logging(self):
        logging.basicConfig(
            level=os.getenv('LOG_LEVEL', 'INFO'),
            format='%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s',
            handlers=[
                logging.FileHandler('/opt/coreflow/logs/bridge.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('CoreBridge')

    def _init_redis(self):
        for attempt in range(3):
            try:
                r = redis.Redis(
                    host=os.getenv('REDIS_HOST'),
                    port=int(os.getenv('REDIS_PORT', int(os.getenv('REDIS_PORT')))),
                    password=os.getenv('REDIS_PASSWORD'),
                    socket_timeout=30
                )
                if r.ping():
                    self.logger.info("✅ Redis verbunden")
                    return r
            except Exception as e:
                self.logger.error(f"Redis Fehler (Versuch {attempt+1}/3): {str(e)}")
                time.sleep(5)
        raise ConnectionError("Redis-Verbindung fehlgeschlagen")

    def _init_zmq(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.PUSH)
        sock.setsockopt(zmq.LINGER, 5000)
        bind_addr = os.getenv('ZMQ_BIND_ADDR', 'tcp://*:int(os.getenv('REDIS_PORT'))')  # Default auf int(os.getenv('REDIS_PORT'))
        sock.bind(bind_addr)
        self.logger.info(f"✅ ZMQ gebunden an {bind_addr}")
        return ctx, sock

    def run(self):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Service gestoppt")
        finally:
            self.zmq_socket.close()
            self.zmq_ctx.term()

if __name__ == "__main__":
    CoreBridge().run()
