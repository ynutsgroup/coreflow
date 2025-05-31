#!/usr/bin/env python3
"""
FTMO Risk Manager Test Suite v2.1

Umfassende Testabdeckung für:
- Risikomanagement-Validierung
- Tägliche Limits
- Trade-Zählung
- Verlustberechnung
- Randfälle und spezielle Szenarien
"""

import unittest
from datetime import datetime, timedelta, timezone
from core.risk_management.ftmo_risk_manager import FTMORiskManager

class TestFTMORiskManager(unittest.TestCase):
    """Testklasse für FTMO Risikomanager"""
    
    def setUp(self):
        """Initialisiert den Risikomanager für jeden Test"""
        self.risk_manager = FTMORiskManager(
            max_daily_trades=5,
            max_risk_per_trade=0.02,
            max_daily_loss=0.05,
            starting_balance=100000.0
        )
    
    def test_initialization(self):
        """Testet die korrekte Initialisierung"""
        self.assertEqual(self.risk_manager.max_daily_trades, 5)
        self.assertEqual(self.risk_manager.max_risk_per_trade, 0.02)
        self.assertEqual(self.risk_manager.max_daily_loss, 0.05)
        self.assertEqual(self.risk_manager.starting_balance, 100000.0)
        self.assertEqual(self.risk_manager.daily_trade_count, 0)
        self.assertEqual(self.risk_manager.daily_loss, 0.0)
    
    def test_trade_validation(self):
        """Testet die Trade-Validierung"""
        # Gültiger Trade
        is_valid, reason = self.risk_manager.validate_trade(500, 0.005)
        self.assertTrue(is_valid)
        self.assertEqual(reason, "Trade validiert")
        
        # Zu hohes Risiko pro Trade
        is_valid, reason = self.risk_manager.validate_trade(3000, 0.03)
        self.assertFalse(is_valid)
        self.assertIn("Risiko 3.0% > Maximum 2.0%", reason)
    
    def test_daily_trade_limit(self):
        """Testet das tägliche Trade-Limit"""
        # 5 Trades durchführen (Maximum)
        for _ in range(5):
            self.risk_manager.register_trade(100, 0.001)
        
        # 6. Trade sollte abgelehnt werden
        is_valid, reason = self.risk_manager.validate_trade(100, 0.001)
        self.assertFalse(is_valid)
        self.assertIn("Max 5 Trades/Tag erreicht", reason)
    
    def test_daily_loss_limit(self):
        """Testet das tägliche Verlustlimit"""
        # 4% Verlust (unter 5% Limit)
        self.risk_manager.register_trade(-4000, -0.04)
        
        # Weitere 2% Verlust (würde 6% ergeben)
        is_valid, reason = self.risk_manager.validate_trade(2000, 0.02)
        self.assertFalse(is_valid)
        self.assertIn("Potentieller Tagesverlust 6000.00 > Maximum 5000.00", reason)
    
    def test_auto_reset(self):
        """Testet das automatische Zurücksetzen bei Tageswechsel"""
        # Simuliere Vortag
        self.risk_manager.last_reset_date = datetime.now(timezone.utc).date() - timedelta(days=1)
        self.risk_manager.daily_trade_count = 5
        self.risk_manager.daily_loss = 4000.0
        
        # Validierung sollte automatisch zurücksetzen
        is_valid, _ = self.risk_manager.validate_trade(100, 0.001)
        self.assertTrue(is_valid)
        self.assertEqual(self.risk_manager.daily_trade_count, 0)
        self.assertEqual(self.risk_manager.daily_loss, 0.0)
    
    def test_trade_registration(self):
        """Testet die Trade-Registrierung"""
        # Gewinn-Trade
        self.risk_manager.register_trade(1500, 0.015)
        self.assertEqual(self.risk_manager.daily_trade_count, 1)
        self.assertEqual(self.risk_manager.daily_loss, 0.0)
        
        # Verlust-Trade
        self.risk_manager.register_trade(-2000, -0.02)
        self.assertEqual(self.risk_manager.daily_trade_count, 2)
        self.assertEqual(self.risk_manager.daily_loss, 2000.0)

    def test_high_balance_trade(self):
        """Testet einen Trade mit hohem Kontostand"""
        risk_manager = FTMORiskManager(starting_balance=1000000.0)
        is_valid, reason = risk_manager.validate_trade(10000, 0.01)  # 1% von 1M = 10K
        self.assertTrue(is_valid)
    
    def test_zero_risk_trade(self):
        """Testet einen Trade mit 0% Risiko"""
        risk_manager = FTMORiskManager()
        is_valid, reason = risk_manager.validate_trade(0, 0.0)
        self.assertTrue(is_valid)
        self.assertEqual(reason, "Trade validiert")

class TestEdgeCases(unittest.TestCase):
    """Testet Grenzfälle und Sonderfälle"""
    
    def test_negative_balance(self):
        """Testet negativen Kontostand"""
        risk_manager = FTMORiskManager(starting_balance=-5000.0)
        is_valid, reason = risk_manager.validate_trade(100, 0.01)
        self.assertFalse(is_valid)
        self.assertIn("Negativer Kontostand", reason)

if __name__ == "__main__":
    # Test-Suite mit detaillierter Ausgabe
    runner = unittest.TextTestRunner(verbosity=2)
    suite = unittest.TestSuite()
    
    # Füge Tests hinzu
    suite.addTest(unittest.makeSuite(TestFTMORiskManager))
    suite.addTest(unittest.makeSuite(TestEdgeCases))
    
    # Führe Tests aus
    print("\n" + "="*60)
    print("FTMO Risk Manager Test Suite")
    print("="*60 + "\n")
    
    result = runner.run(suite)
    
    # Zusammenfassung
    print("\nTest-Zusammenfassung:")
    print(f"Gesamt: {result.testsRun}")
    print(f"Erfolgreich: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Fehlgeschlagen: {len(result.failures)}")
    print(f"Fehler: {len(result.errors)}")
    
    if not result.wasSuccessful():
        raise SystemExit(1)
