#!/usr/bin/env python3
"""
CoreFlow Bridge Service - Auto .env Configuration
"""

import os
import time
import logging
import redis
import zmq
from pathlib import Path
from cryptography.fernet import Fernet
from dotenv import load_dotenv

# Auto-load .env from standard locations
load_dotenv('/opt/coreflow/.env', override=True)

class BridgeService:
    def __init__(self):
        """Initialize with .env configuration"""
        self.logger = self._setup_logging()
        self.redis = self._connect_redis()
        self.zmq_context, self.socket = self._setup_zmq()
        self.logger.info("Service initialized with .env config")

    def _setup_logging(self):
        """Configure logging from .env"""
        logging.basicConfig(
            level=os.getenv('LOG_LEVEL', 'INFO'),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/opt/coreflow/logs/bridge.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    def _connect_redis(self):
        """Auto-configured Redis connection"""
        while True:
            try:
                r = redis.Redis(
                    host=os.getenv('REDIS_HOST'),
                    port=int(os.getenv('REDIS_PORT', int(os.getenv('REDIS_PORT')))),
                    password=os.getenv('REDIS_PASSWORD'),
                    ssl=os.getenv('REDIS_SSL', 'false').lower() == 'true',
                    socket_timeout=10
                )
                r.ping()
                self.logger.info("Redis connection established")
                return r
            except Exception as e:
                self.logger.error(f"Redis connection failed: {str(e)}")
                time.sleep(5)

    def _setup_zmq(self):
        """ZeroMQ setup with .env config"""
        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        socket.bind(os.getenv('ZMQ_BIND_ADDR', 'tcp://*:int(os.getenv('REDIS_PORT'))'))
        self.logger.info(f"ZMQ bound to {os.getenv('ZMQ_BIND_ADDR')}")
        return context, socket

    def run(self):
        """Main service loop"""
        try:
            while True:
                # Your message processing here
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Service stopped by user")
        finally:
            self.socket.close()
            self.zmq_context.term()

if __name__ == "__main__":
    BridgeService().run()
