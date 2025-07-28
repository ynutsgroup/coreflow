
import os

def calculate_lot_size(account_balance, stop_loss_pips, risk_percent, pip_value_per_lot=10):
    """
    Berechnet Lot-Größe basierend auf Risiko-Parametern.
    """
    risk_amount = account_balance * (risk_percent / 100)
    lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)
    return round(lot_size, 2)

def is_ftmo_compliant(daily_loss, max_daily_loss, total_drawdown, max_drawdown):
    """
    Prüft, ob Trade nach FTMO-Regeln erlaubt ist.
    """
    if daily_loss < -abs(max_daily_loss):
        print("❌ Daily Loss Limit überschritten")
        return False
    if total_drawdown < -abs(max_drawdown):
        print("❌ Gesamt-Drawdown überschritten")
        return False
    return True

# Beispiel-Nutzung
if __name__ == "__main__":
    balance = 100000  # FTMO Challenge-Konto
    sl = 30           # Stop-Loss in Pips
    risk = 1.0        # 1% Risiko pro Trade

    lot = calculate_lot_size(balance, sl, risk)
    print(f"Erlaubte Lot-Größe: {lot} Lots")

    # FTMO-Regeln prüfen
    if is_ftmo_compliant(-4900, -5000, -9500, -10000):
        print("✅ Trade erlaubt (FTMO konform)")
    else:
        print("❌ Kein Trade (Regel verletzt)")
