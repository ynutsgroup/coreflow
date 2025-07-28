#!/usr/bin/env python3
"""
QuantumFlow Commander Pro v2.1
- Integrierte Telegram-Steuerung
- FTMO-konformes Risikomanagement
- Multi-Thread-Sicherheit
"""

import os
import threading
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext

# FTMO Risikomanagement (Modular)
class FTMO_Risk_Engine:
    RISK_PROFILES = {
        'conservative': {'max_daily': 0.01, 'max_trade': 0.02},
        'moderate': {'max_daily': 0.02, 'max_trade': 0.05},
        'aggressive': {'max_daily': 0.03, 'max_trade': 0.08}
    }
    
    def __init__(self):
        self.current_profile = 'moderate'
        
    def set_risk(self, profile):
        if profile in self.RISK_PROFILES:
            self.current_profile = profile
            return True
        return False

# Haupt-Trading-System
class QuantumFlowSystem:
    def __init__(self):
        self.risk_engine = FTMO_Risk_Engine()
        self.live_trading = False
        self.thread_lock = threading.Lock()
        
    def get_status(self):
        with self.thread_lock:
            return {
                'risk_profile': self.risk_engine.current_profile,
                'balance': 1_000_000,  # Mock
                'running': self.live_trading
            }

# Telegram Controller
class QuantumCommander:
    def __init__(self, token, trading_system):
        self.bot = Updater(token=token, use_context=True)
        self.ts = trading_system
        
        # Befehle registrieren
        dp = self.bot.dispatcher
        dp.add_handler(CommandHandler("start", self.cmd_start))
        dp.add_handler(CommandHandler("risk", self.cmd_risk))
        dp.add_handler(CommandHandler("status", self.cmd_status))
        dp.add_handler(CommandHandler("stop", self.cmd_stop))
        
    def cmd_start(self, update: Update, context: CallbackContext):
        """Startsystem mit Risikoprofil"""
        args = context.args
        if len(args) > 0 and self.ts.risk_engine.set_risk(args[0]):
            self.ts.live_trading = True
            update.message.reply_text(
                f"🚀 QuantumFlow gestartet ({args[0]} Risiko)\n"
                f"Max Trade: {self.ts.risk_engine.RISK_PROFILES[args[0]]['max_trade']*100}%"
            )
        else:
            update.message.reply_text("⚠️ Nutzung: /start [conservative|moderate|aggressive]")

    def cmd_risk(self, update: Update, context: CallbackContext):
        """Risikoprofil dynamisch anpassen"""
        if len(context.args) == 0:
            profiles = "\n".join(self.ts.risk_engine.RISK_PROFILES.keys())
            update.message.reply_text(f"Verfügbare Profile:\n{profiles}")
            return
            
        profile = context.args[0].lower()
        with self.ts.thread_lock:
            success = self.ts.risk_engine.set_risk(profile)
        
        if success:
            update.message.reply_text(f"✅ Risiko auf {profile} gesetzt")
        else:
            update.message.reply_text("❌ Ungültiges Profil")

    def cmd_status(self, update: Update, context: CallbackContext):
        """Systemstatus abfragen"""
        status = self.ts.get_status()
        update.message.reply_text(
            f"📊 QuantumFlow Status\n"
            f"• Risiko: {status['risk_profile']}\n"
            f"• Kontostand: ${status['balance']:,.2f}\n"
            f"• Live: {'✅' if status['running'] else '❌'}"
        )

    def cmd_stop(self, update: Update, context: CallbackContext):
        """Trading anhalten"""
        with self.ts.thread_lock:
            self.ts.live_trading = False
        update.message.reply_text("🛑 Trading gestoppt")

    def start(self):
        self.bot.start_polling()
        self.bot.idle()

# Initialisierung
if __name__ == "__main__":
    # Konfiguration laden
    load_dotenv()
    TOKEN = os.getenv("TELEGRAM_TOKEN")
    ADMIN_IDS = [int(id) for id in os.getenv("ADMIN_IDS").split(",")]
    
    # Systeme starten
    qf = QuantumFlowSystem()
    commander = QuantumCommander(TOKEN, qf)
    
    # Threads starten
    trading_thread = threading.Thread(target=qf.run)
    bot_thread = threading.Thread(target=commander.start)
    
    trading_thread.start()
    bot_thread.start()
    
    print("""
    ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗██╗   ██╗██╗
    ██╔══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██║   ██║██║
    ██████╔╝██║   ██║███████║██╔██╗ ██║   ██║   ██║   ██║██║
    ██╔═══╝ ██║   ██║██╔══██║██║╚██╗██║   ██║   ██║   ██║██║
    ██║     ╚██████╔╝██║  ██║██║ ╚████║   ██║   ╚██████╔╝██║
    ╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═╝
    """)
    print("System aktiv! Telegram-Befehle:")
    print("/start [profil] - Trading starten")
    print("/risk [profil] - Risiko anpassen")
    print("/status - Systeminfo")
    print("/stop - Trading anhalten")
