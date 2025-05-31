#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FTMO-Compliant Telegram Notification Module for CoreFlow
Enhanced with trading rule alerts and message formatting
"""

import os
import requests
import logging
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional

# Initialize logging
logger = logging.getLogger("CF.Telegram")

load_dotenv("/opt/coreflow/.env")

class TelegramNotifier:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.timeout = 10
        self.last_alert = None
        self.alert_cooldown = 60  # seconds
        
        if not self.token or not self.chat_id:
            logger.error("Telegram credentials missing in .env")

    def send_alert(self, message: str, alert_type: Optional[str] = "INFO") -> bool:
        """
        FTMO-compliant alerting with message formatting and rate limiting
        Types: INFO, WARNING, ERROR, TRADE, RISK
        """
        if not self.token or not self.chat_id:
            return False

        # Rate limiting check
        if self.last_alert and (datetime.now() - self.last_alert).seconds < self.alert_cooldown:
            logger.debug("Alert cooldown active")
            return False

        # FTMO message formatting
        formatted_msg = self._format_message(message, alert_type)
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        
        payload = {
            "chat_id": self.chat_id,
            "text": formatted_msg,
            "parse_mode": "HTML"
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            if response.status_code == 200:
                self.last_alert = datetime.now()
                return True
                
            logger.error(f"Telegram API error: {response.status_code} - {response.text}")
            return False
            
        except Exception as e:
            logger.error(f"Telegram send failed: {str(e)}")
            return False

    def _format_message(self, message: str, alert_type: str) -> str:
        """Formats messages according to FTMO notification standards"""
        icons = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "TRADE": "💱", 
            "RISK": "🚨"
        }
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        icon = icons.get(alert_type, "🔔")
        
        return f"<b>{icon} {alert_type}</b>\n{timestamp}\n\n{message}"

# Global instance for easy access
notifier = TelegramNotifier()

def send_telegram_alert(message: str, alert_type: Optional[str] = "INFO") -> bool:
    """Legacy function wrapper for backward compatibility"""
    return notifier.send_alert(message, alert_type)
if __name__ == "__main__":
    send_telegram_alert("✅ Telegram Modul erfolgreich getestet!", "INFO")
