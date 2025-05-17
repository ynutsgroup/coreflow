#!/usr/bin/env python3
# CoreFlow Pro Trading System v5.1 (CLI Version)

import argparse
import os
import sys
import time
import logging
from dotenv import load_dotenv

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CoreFlowPro")

# --- CLI Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description='CoreFlow Pro Trading System')
    parser.add_argument('--ftmo', action='store_true', help='Enable FTMO compliance rules')
    parser.add_argument('--risk', type=float, default=1.0, help='Maximum risk percentage per trade')
    parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration')
    return parser.parse_args()

# --- Main Trading Engine ---
class TradingEngine:
    def __init__(self, args):
        self.args = args
        self.config = self.load_config()

        if args.ftmo:
            self.apply_ftmo_rules()

        if args.gpu:
            self.init_gpu()

        logger.info(f"Starting with Risk: {self.config['risk']}% | FTMO: {args.ftmo} | GPU: {args.gpu}")

    def load_config(self):
        """Load configuration from .env with CLI priority"""
        load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

        return {
            'risk': self.args.risk or float(os.getenv('MAX_RISK_PERCENT', 1.0)),
            'ftmo': self.args.ftmo or os.getenv('FTMO_ACCOUNT', 'false').lower() == 'true',
            'gpu': self.args.gpu and self.check_gpu_support()
        }

    def apply_ftmo_rules(self):
        """Enforce FTMO-specific trading rules"""
        self.ftmo_rules = {
            'max_daily_loss': 0.05,  # 5%
            'min_holding_time': 300,  # 5 minutes
            'prohibited_symbols': ['USOIL', 'UKOIL', 'XAUUSDm']
        }
        logger.info("FTMO compliance rules activated")

    def init_gpu(self):
        """Initialize GPU support if available"""
        try:
            import torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"GPU Acceleration: {self.device}")
        except ImportError:
            logger.warning("GPU support requires 'torch' package. Falling back to CPU.")
            self.device = 'cpu'

    def check_gpu_support(self):
        """Check if GPU is available and torch is installed"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def run(self):
        """Main trading loop"""
        try:
            while True:
                # Trading logic placeholder
                logger.info("Running trading logic...")
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutdown requested. Exiting gracefully.")

# --- Entry Point ---
if __name__ == "__main__":
    args = parse_args()

    if not (0.1 <= args.risk <= 5.0):
        print("Error: Risk must be between 0.1% and 5.0%")
        sys.exit(1)

    engine = TradingEngine(args)
    engine.run()
