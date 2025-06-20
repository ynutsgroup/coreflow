#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coreflow_captain.py – Advanced CoreFlow Security and Control Module

Enhanced Features:
- File fingerprinting with SHA-256
- Recursive directory scanning
- Comprehensive verification with metadata checks
- Secure fingerprint storage options
- Rate-limited Telegram alerts
- Configurable monitoring modes
- Detailed logging and reporting

Standort: /opt/coreflow/utils/
"""

import os
import sys
import json
import hashlib
import logging
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import requests

# === Configuration Class ===
class Config:
    """Centralized configuration management"""
    def __init__(self):
        load_dotenv('/opt/coreflow/.env')
        self.TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
        self.TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
        self.FINGERPRINT_DIR = os.getenv("FINGERPRINT_DIR", "/opt/coreflow/fingerprints/")
        self.ALERT_RATE_LIMIT = int(os.getenv("ALERT_RATE_LIMIT", 60))  # seconds
        self.LOG_FILE = os.getenv("LOG_FILE", "/var/log/coreflow/captain.log")
        
        # Ensure fingerprint directory exists
        os.makedirs(self.FINGERPRINT_DIR, exist_ok=True)
        
        # Telegram icons mapping
        self.ICONS = {
            "info": "📘", 
            "success": "✅", 
            "warning": "⚠️", 
            "error": "❌",
            "critical": "🚨"
        }

# Initialize config
config = Config()

# === Advanced Logging ===
class CustomFormatter(logging.Formatter):
    """Custom log formatter with colors"""
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = '%(asctime)s | %(levelname)s | %(message)s'

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

logger = logging.getLogger("CoreFlowCaptain")
logger.setLevel(logging.INFO)

# Console handler
ch = logging.StreamHandler()
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

# File handler
fh = logging.FileHandler(config.LOG_FILE)
fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
logger.addHandler(fh)

# === Security Functions ===
class FileIntegrity:
    """Handles all file integrity operations"""
    
    @staticmethod
    def get_file_metadata(filepath: str) -> Dict:
        """Get comprehensive file metadata"""
        stat = os.stat(filepath)
        return {
            "size": stat.st_size,
            "permissions": stat.st_mode,
            "owner": stat.st_uid,
            "group": stat.st_gid,
            "last_modified": stat.st_mtime,
            "created": stat.st_ctime
        }
    
    @staticmethod
    def generate_fingerprint(filepath: str) -> Dict:
        """Generate complete file fingerprint"""
        try:
            with open(filepath, "rb") as f:
                content = f.read()
            
            return {
                "file": os.path.basename(filepath),
                "path": os.path.abspath(filepath),
                "sha256": hashlib.sha256(content).hexdigest(),
                "metadata": FileIntegrity.get_file_metadata(filepath),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "system": {
                    "hostname": os.uname().nodename,
                    "python_version": sys.version
                }
            }
        except Exception as e:
            logger.error(f"Failed to generate fingerprint for {filepath}: {str(e)}")
            raise

    @staticmethod
    def get_fingerprint_path(filepath: str) -> str:
        """Get secure fingerprint file path"""
        file_hash = hashlib.md5(filepath.encode()).hexdigest()
        return os.path.join(config.FINGERPRINT_DIR, f"{file_hash}.fingerprint.json")

    @classmethod
    def sign_file(cls, filepath: str) -> bool:
        """Create and store file fingerprint"""
        try:
            fingerprint = cls.generate_fingerprint(filepath)
            out_path = cls.get_fingerprint_path(filepath)
            
            with open(out_path, "w") as f:
                json.dump(fingerprint, f, indent=2)
            
            logger.info(f"Fingerprint created for {filepath} at {out_path}")
            return True
        except Exception as e:
            logger.error(f"Error signing file {filepath}: {str(e)}")
            return False

    @classmethod
    def verify_file(cls, filepath: str) -> Tuple[bool, Dict]:
        """Verify file against stored fingerprint with detailed results"""
        fingerprint_file = cls.get_fingerprint_path(filepath)
        results = {
            "file": filepath,
            "passed": False,
            "checks": {
                "fingerprint_exists": False,
                "content_match": False,
                "metadata_match": False,
                "permissions_match": False
            },
            "differences": []
        }

        if not os.path.exists(fingerprint_file):
            results["differences"].append("No fingerprint file exists")
            return (False, results)
        
        results["checks"]["fingerprint_exists"] = True

        try:
            # Load saved fingerprint
            with open(fingerprint_file, "r") as f:
                saved = json.load(f)
            
            # Generate current fingerprint
            current = cls.generate_fingerprint(filepath)
            
            # Compare content hash
            if current["sha256"] == saved["sha256"]:
                results["checks"]["content_match"] = True
            else:
                results["differences"].append("File content has changed")
            
            # Compare metadata
            metadata_checks = [
                ("size", "File size"),
                ("permissions", "File permissions"),
                ("owner", "File owner"),
                ("group", "File group")
            ]
            
            for field, description in metadata_checks:
                if current["metadata"][field] != saved["metadata"][field]:
                    results["differences"].append(
                        f"{description} changed from {saved['metadata'][field]} to {current['metadata'][field]}"
                    )
                else:
                    if field == "permissions":
                        results["checks"]["permissions_match"] = True
                    else:
                        results["checks"]["metadata_match"] = True
            
            # Determine overall status
            results["passed"] = all(results["checks"].values())
            
            return (results["passed"], results)
            
        except Exception as e:
            logger.error(f"Verification failed for {filepath}: {str(e)}")
            results["differences"].append(f"Verification error: {str(e)}")
            return (False, results)

# === Alert System ===
class AlertSystem:
    """Handles all alerting functionality with rate limiting"""
    
    _last_alert_time = {}
    
    @classmethod
    def send_telegram(cls, message: str, level: str = "info") -> bool:
        """Send Telegram alert with rate limiting"""
        if not config.TELEGRAM_TOKEN or not config.TELEGRAM_CHAT_ID:
            logger.warning("Telegram not configured")
            return False
            
        # Rate limiting check
        current_time = time.time()
        last_time = cls._last_alert_time.get(level, 0)
        
        if current_time - last_time < config.ALERT_RATE_LIMIT:
            logger.debug(f"Rate limited {level} alert: {message[:50]}...")
            return False
            
        try:
            icon = config.ICONS.get(level, "ℹ️")
            url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
            payload = {
                "chat_id": config.TELEGRAM_CHAT_ID,
                "text": f"{icon} {message}",
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            cls._last_alert_time[level] = current_time
            logger.info(f"Sent {level} alert to Telegram")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Telegram send failed: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected Telegram error: {str(e)}")
            return False
    
    @classmethod
    def alert_file_change(cls, filepath: str, results: Dict) -> None:
        """Generate comprehensive alert for file changes"""
        if results["passed"]:
            return
            
        alert_lines = [
            f"*File Integrity Alert*",
            f"`{os.path.basename(filepath)}`",
            "",
            "*Changes detected:*",
        ]
        
        alert_lines.extend(results["differences"])
        
        alert_lines.extend([
            "",
            "*Verification results:*",
            f"- Fingerprint exists: {'✅' if results['checks']['fingerprint_exists'] else '❌'}",
            f"- Content matches: {'✅' if results['checks']['content_match'] else '❌'}",
            f"- Metadata matches: {'✅' if results['checks']['metadata_match'] else '❌'}",
            f"- Permissions match: {'✅' if results['checks']['permissions_match'] else '❌'}",
            "",
            f"*Path:* `{filepath}`",
            f"*Host:* `{os.uname().nodename}`"
        ])
        
        message = "\n".join(alert_lines)
        cls.send_telegram(message, "warning" if results["checks"]["fingerprint_exists"] else "error")

# === Directory Scanner ===
class DirectoryScanner:
    """Recursive directory scanner for monitoring"""
    
    @staticmethod
    def find_files_to_monitor(base_path: str, extensions: Optional[List[str]] = None) -> List[str]:
        """Find all files matching criteria in directory tree"""
        if not os.path.exists(base_path):
            logger.error(f"Base path does not exist: {base_path}")
            return []
            
        matched_files = []
        
        for root, _, files in os.walk(base_path):
            for file in files:
                if extensions:
                    if any(file.endswith(ext) for ext in extensions):
                        matched_files.append(os.path.join(root, file))
                else:
                    matched_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(matched_files)} files in {base_path}")
        return matched_files

# === CLI Interface ===
def setup_argparse() -> argparse.ArgumentParser:
    """Configure command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="CoreFlow Captain - File Integrity Monitoring System",
        epilog="Example: ./coreflow_captain.py monitor --path /etc --extensions .conf,.cfg"
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Sign command
    sign_parser = subparsers.add_parser("sign", help="Create fingerprint for file")
    sign_parser.add_argument("path", help="File path to fingerprint")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify file against fingerprint")
    verify_parser.add_argument("path", help="File path to verify")
    verify_parser.add_argument("--no-alert", action="store_true", help="Skip sending alerts")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor directory for changes")
    monitor_parser.add_argument("--path", required=True, help="Directory to monitor")
    monitor_parser.add_argument("--extensions", help="Comma-separated file extensions to monitor")
    monitor_parser.add_argument("--interval", type=int, default=300, 
                               help="Monitoring interval in seconds")
    
    # Batch sign command
    batch_parser = subparsers.add_parser("batch-sign", help="Create fingerprints for multiple files")
    batch_parser.add_argument("--path", required=True, help="Directory to process")
    batch_parser.add_argument("--extensions", help="Comma-separated file extensions to include")
    
    return parser

def main():
    """Main entry point for CLI"""
    parser = setup_argparse()
    args = parser.parse_args()
    
    try:
        if args.command == "sign":
            if FileIntegrity.sign_file(args.path):
                logger.info(f"Successfully signed {args.path}")
            else:
                logger.error(f"Failed to sign {args.path}")
                sys.exit(1)
                
        elif args.command == "verify":
            passed, results = FileIntegrity.verify_file(args.path)
            if passed:
                logger.info(f"Verification passed for {args.path}")
            else:
                logger.warning(f"Verification failed for {args.path}")
                print(json.dumps(results, indent=2))
                
                if not args.no_alert:
                    AlertSystem.alert_file_change(args.path, results)
                sys.exit(1)
                
        elif args.command == "monitor":
            extensions = args.extensions.split(",") if args.extensions else None
            logger.info(f"Starting monitoring of {args.path} (interval: {args.interval}s)")
            
            try:
                while True:
                    files = DirectoryScanner.find_files_to_monitor(args.path, extensions)
                    for file in files:
                        passed, results = FileIntegrity.verify_file(file)
                        if not passed:
                            AlertSystem.alert_file_change(file, results)
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                
        elif args.command == "batch-sign":
            extensions = args.extensions.split(",") if args.extensions else None
            files = DirectoryScanner.find_files_to_monitor(args.path, extensions)
            
            success = 0
            for file in files:
                if FileIntegrity.sign_file(file):
                    success += 1
            
            logger.info(f"Batch signing complete: {success}/{len(files)} files processed successfully")
            if success != len(files):
                sys.exit(1)
                
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        AlertSystem.send_telegram(f"Captain crashed: {str(e)}", "critical")
        sys.exit(1)

if __name__ == "__main__":
    main()
