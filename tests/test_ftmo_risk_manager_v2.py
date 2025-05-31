#!/usr/bin/env python3
"""
FTMO Risk Manager Test Suite v3.0

Umfassende Tests für:
- Grenzfälle und Extremwerte
- FTMO-Regelkonforme Validierung
- Detailliertes Logging
"""

import unittest
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
import asyncio
from telegram import Bot
from core.risk_management.ftmo_risk_manager import get_ftmo_risk_manager

# Telegram-Bot senden Nachricht (async)
async def send_telegram_message(message: str):
    bot = Bot(token="DEIN_BOT_TOKEN")  # Dein Bot-Token hier einfügen
    chat_id = "DEIN_CHAT_ID"  # Deine Chat-ID hier einfügen
    await bot.send_message(chat_id=chat_id, text=message)

class TestFTMORiskManager(unittest.TestCase):
    """Haupttestklasse für FTMO Risikomanager"""
    
    @classmethod
    def setUpClass(cls):
        """Einmalige Initialisierung für alle Tests"""
        cls.logger = logging.getLogger('FTMOTests')
        cls.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        cls.logger.addHandler(handler)
    
    def setUp(self):
        """Initialisierung vor jedem Test"""
        self.manager = get_ftmo_risk_manager()
        self.manager.config = {
            'max_daily_trades': 5,
            'max_risk_per_trade': 0.02,
            'max_daily_loss': 0.05,
            'starting_balance': 100000.0,
            'instrument_rules': {
                'crypto': {'max_risk_multiplier': 0.5},
                'forex': {'max_risk_multiplier': 1.0}
            }
        }
        self.manager.force_reset()
        self.logger.info("\n" + "="*50 + f"\nStarting test: {self._testMethodName}\n" + "="*50)
    
    def tearDown(self):
        """Aufräumen nach jedem Test"""
        self.manager.force_reset()

    def test_grenzfaelle(self):
        """Testet Extremwerte und Grenzfälle"""
        test_cases = [
            (0, 0.0, True, "Null-Risiko Trade"),
            (1, 0.000001, True, "Minimales Risiko"),
            (-1, -0.000001, True, "Minimaler Verlust"),
            (50000, 0.02, True, "Maximales Risiko pro Trade"),
            (-5000, -0.05, True, "Maximaler Tagesverlust"),
            (50001, 0.020001, False, "Über maximalem Risiko pro Trade"),
            (-5001, -0.050001, False, "Über maximalem Tagesverlust"),
            (float('inf'), 1.0, False, "Unendlicher Wert"),
            (1, float('nan'), False, "Ungültiges Risiko (NaN)"),
            (1, None, False, "None als Risiko")
        ]
        
        for value, risk, expected, desc in test_cases:
            with self.subTest(desc=desc):
                valid, _ = self.manager.validate_trade(value, risk)
                self.assertEqual(valid, expected, f"Fehler bei: {desc}")

    def test_ftmo_spezifische_regeln(self):
        """Testet FTMO-spezifische Regeln"""
        valid, msg = self.manager.validate_trade(30000, 0.02, instrument='crypto')
        self.assertFalse(valid)
        self.assertIn("exceeds max", msg)
        
        valid, msg = self.manager.validate_trade(30000, 0.02, instrument='forex')
        self.assertTrue(valid)
        
        self.manager.register_trade(-2500, -0.025)
        self.manager.register_trade(-2000, -0.02)
        valid, msg = self.manager.validate_trade(-501, -0.005)
        self.assertFalse(valid)
        self.assertIn("exceeds max", msg)

    def test_detailed_logging(self):
        """Testet detailliertes Logging"""
        with self.assertLogs('FTMORiskManager', level='INFO') as log:
            self.manager.register_trade(1500, 0.015)
            self.assertIn("Trade registered", log.output[0])
            
            self.manager.register_trade(-2000, -0.02)
            self.assertIn("Daily P/L", log.output[1])

    def test_komplexe_szenarien(self):
        """Testet komplexe Handels-Szenarien"""
        for i in range(3):
            self.manager.register_trade(1000, 0.01)
            self.assertEqual(self.manager.daily_trade_count, i+1)
        
        self.manager.register_trade(-3000, -0.03)
        stats = self.manager.get_daily_stats()
        self.assertAlmostEqual(stats['net_pnl'], 0.0, places=2)
        
        self.manager.force_reset()
        self.assertEqual(self.manager.daily_trade_count, 0)

    def test_send_message_to_telegram(self):
        """Testet, ob eine Nachricht an Telegram gesendet wird"""
        # Wir können die Telegram-Funktion testen
        message = "Testnachricht an Telegram"
        asyncio.run(send_telegram_message(message))  # sendet die Nachricht
        self.assertTrue(True)  # Wenn keine Ausnahme auftritt, war der Test erfolgreich

class TestEdgeCases(unittest.TestCase):
    """Spezielle Grenzfalltests"""
    
    def setUp(self):
        self.manager = get_ftmo_risk_manager()
        self.manager.force_reset()
    
    def test_minimalwerte(self):
        valid, _ = self.manager.validate_trade(0.01, 0.0000001)
        self.assertTrue(valid)
        
        valid, _ = self.manager.validate_trade(-0.01, -0.0000001)
        self.assertTrue(valid)
    
    def test_rounding_errors(self):
        for _ in range(1000):
            self.manager.register_trade(0.00001, 0.0000001)
        
        stats = self.manager.get_daily_stats()
        self.assertAlmostEqual(stats['daily_profit'], 0.01, places=4)

def create_test_suite():
    """Erstellt die Testsuite für die Ausführung"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestFTMORiskManager))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    return suite

def main():
    """Test Runner"""
    loader = unittest.TestLoader()
    suite = create_test_suite()
    
    runner = unittest.TextTestRunner(
        verbosity=2,
        failfast=True,
        buffer=True
    )
    
    print("\n" + "="*60)
    print("FTMO RISK MANAGER KOMPLETTTESTSUITE v3.0")
    print("="*60 + "\n")
    
    result = runner.run(suite)
    
    print("\n" + "="*60)
    print("TESTZUSAMMENFASSUNG")
    print("="*60)
    print(f"Tests insgesamt: {result.testsRun}")
    print(f"Erfolgreich: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Fehlgeschlagen: {len(result.failures)}")
    print(f"Fehler: {len(result.errors)}")
    
    if result.failures:
        print("\nFEHLERDETAILS:")
        for failure in result.failures:
            print(f"\n{failure[0]._testMethodName}:")
            print(failure[1])
    
    if not result.wasSuccessful():
        raise SystemExit(1)

if __name__ == "__main__":
    main()
