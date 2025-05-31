import os
import logging
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple
from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from dotenv import load_dotenv

# Load configuration
load_dotenv()

# Configuration with type hints and defaults
CONFIG = {
    'TELEGRAM_TOKEN': os.getenv("TELEGRAM_TOKEN"),
    'TELEGRAM_CHAT_ID': os.getenv("TELEGRAM_CHAT_ID"),
    'MAX_DAILY_LOSS': float(os.getenv("MAX_DAILY_LOSS", "0.05")),
    'MAX_OVERALL_LOSS': float(os.getenv("MAX_OVERALL_LOSS", "0.10")),
    'MAX_DAILY_TRADES': int(os.getenv("MAX_DAILY_TRADES", "5")),
    'MAX_RISK_PER_TRADE': float(os.getenv("MAX_RISK_PER_TRADE", "0.02")),
    'STARTING_BALANCE': float(os.getenv("STARTING_BALANCE", "100000.0")),
    'ALERT_THRESHOLD': float(os.getenv("ALERT_THRESHOLD", "0.75"))  # 75% of limit
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ftmo_risk.log')
    ]
)
logger = logging.getLogger('FTMO_RiskManager')

class FTMO_RiskManager:
    def __init__(self):
        """Initialize the risk manager with configuration"""
        self._validate_config()
        self.daily_trade_count = 0
        self.daily_loss = 0.0
        self.daily_profit = 0.0
        self.overall_loss = 0.0
        self.last_reset_date = datetime.now(timezone.utc).date()

    def _validate_config(self):
        """Validate configuration parameters"""
        if not CONFIG['TELEGRAM_TOKEN'] or not CONFIG['TELEGRAM_CHAT_ID']:
            raise ValueError("Telegram credentials not configured")
        if CONFIG['MAX_DAILY_LOSS'] <= 0 or CONFIG['MAX_OVERALL_LOSS'] <= 0:
            raise ValueError("Loss limits must be positive")
        if CONFIG['STARTING_BALANCE'] <= 0:
            raise ValueError("Starting balance must be positive")

    def _setup(self):
        """Initial setup tasks"""
        logger.info("FTMO Risk Manager initialized with config: %s", {
            k: v for k, v in CONFIG.items() if k not in ['TELEGRAM_TOKEN']
        })

    async def _send_telegram_alert(self, message: str) -> bool:
        """Send a Telegram alert with proper error handling"""
        try:
            async with Bot(token=CONFIG['TELEGRAM_TOKEN']) as bot:
                await bot.send_message(
                    chat_id=CONFIG['TELEGRAM_CHAT_ID'],
                    text=f"🚨 FTMO Alert:\n{message}",
                    parse_mode="Markdown"
                )
            logger.info("Telegram alert sent: %s", message)
            return True
        except TelegramAPIError as e:
            logger.error("Telegram API error: %s", str(e))
        except Exception as e:
            logger.error("Unexpected error sending alert: %s", str(e))
        return False

    def _auto_reset(self):
        """Auto-reset daily counters at UTC midnight"""
        today = datetime.now(timezone.utc).date()
        if self.last_reset_date != today:
            self.reset_counts()

    def reset_counts(self):
        """Reset daily counters"""
        self.daily_trade_count = 0
        self.daily_loss = 0.0
        self.daily_profit = 0.0
        self.last_reset_date = datetime.now(timezone.utc).date()
        logger.info("Daily counters reset")

    async def validate_trade(self, trade_value: float, risk_percentage: float) -> Tuple[bool, str]:
        """Validate a trade against FTMO rules"""
        self._auto_reset()

        # Validate trade parameters
        if not isinstance(trade_value, (int, float)):
            return False, "Invalid trade value"
        if not isinstance(risk_percentage, (int, float)) or abs(risk_percentage) > 1:
            return False, "Risk percentage must be between -1 and 1"

        # Check daily trade limit
        if self.daily_trade_count >= CONFIG['MAX_DAILY_TRADES']:
            msg = f"Max daily trades ({CONFIG['MAX_DAILY_TRADES']}) reached"
            await self._send_telegram_alert(msg)
            return False, msg

        # Check risk per trade
        if abs(risk_percentage) > CONFIG['MAX_RISK_PER_TRADE']:
            msg = f"Risk {abs(risk_percentage)*100:.2f}% exceeds max {CONFIG['MAX_RISK_PER_TRADE']*100:.2f}%"
            await self._send_telegram_alert(msg)
            return False, msg

        # Check daily loss limit
        if trade_value < 0:
            potential_loss = self.daily_loss + trade_value
            max_daily_loss = CONFIG['MAX_DAILY_LOSS'] * CONFIG['STARTING_BALANCE']
            
            if abs(potential_loss) > max_daily_loss:
                msg = f"Potential daily loss {abs(potential_loss):.2f} exceeds max {max_daily_loss:.2f}"
                await self._send_telegram_alert(msg)
                return False, msg

            # Check alert threshold (75% of limit by default)
            if abs(potential_loss) > max_daily_loss * CONFIG['ALERT_THRESHOLD']:
                await self._send_telegram_alert(
                    f"⚠️ Approaching daily loss limit: {abs(potential_loss)/max_daily_loss:.0%}"
                )

        return True, "Trade validated"

    async def register_trade(self, trade_value: float, risk_percentage: float) -> bool:
        """Register a new trade with validation"""
        is_valid, reason = await self.validate_trade(trade_value, risk_percentage)
        if not is_valid:
            logger.warning("Trade rejected: %s", reason)
            return False

        self.daily_trade_count += 1
        if trade_value < 0:
            self.daily_loss += trade_value
            self.overall_loss += trade_value
        else:
            self.daily_profit += trade_value

        logger.info(
            "Trade registered: Value %.2f | Risk %.2f%% | Daily P/L %.2f",
            trade_value, risk_percentage * 100, self.daily_profit + self.daily_loss
        )
        return True

    def get_daily_stats(self) -> Dict[str, float]:
        """Get current daily statistics"""
        return {
            'trades': self.daily_trade_count,
            'loss': self.daily_loss,
            'profit': self.daily_profit,
            'net': self.daily_profit + self.daily_loss,
            'remaining_trades': max(0, CONFIG['MAX_DAILY_TRADES'] - self.daily_trade_count),
            'remaining_daily_loss': max(0, CONFIG['MAX_DAILY_LOSS'] * CONFIG['STARTING_BALANCE'] + self.daily_loss)
        }

    async def close(self):
        """Clean up resources"""
        await self.bot.close()

async def main():
    """Example usage"""
    risk_manager = FTMO_RiskManager()
    try:
        # Example trades
        await risk_manager.register_trade(1500, 0.01)  # Valid trade
        await risk_manager.register_trade(-2000, -0.02)  # Loss trade
        
        # Try invalid trade
        await risk_manager.register_trade(-50000, -0.05)  # Should fail
        
        # Show stats
        logger.info("Daily stats: %s", risk_manager.get_daily_stats())
    finally:
        await risk_manager.close()

if __name__ == "__main__":
    asyncio.run(main())
