def calculate_lot_size(
    balance: float,
    risk_percent: float,
    stoploss_pips: float,
    pip_value: float = 10.0,
    symbol: str = None
) -> float:
    """
    Berechnet FTMO-konforme Lot-Größe mit strengen Risikolimits.
    
    Args:
        balance: Kontostand in Kontowährung (muss > 0 sein)
        risk_percent: Risiko in % pro Trade (0.1-1.0% für FTMO)
        stoploss_pips: SL in Pips (> 0.5 für FTMO)
        pip_value: Wert pro Pip in Kontowährung
        symbol: Optionales Symbol für positionsgrößenabhängige Limits
        
    Returns:
        Lot-Größe auf 2 Dezimalstellen gerundet
        
    Raises:
        ValueError: Bei Regelverstoß
        
    FTMO-Regeln:
        - Max 1% Risiko pro Trade
        - Min 0.5 Pips SL-Distanz
        - Max 50 Lots pro Position
    """
    # Input-Validierung
    if balance <= 0:
        raise ValueError(f"Kontostand muss positiv sein: {balance}")
    if not 0.1 <= risk_percent <= 1.0:
        raise ValueError(f"FTMO Risiko muss 0.1-1.0% sein: {risk_percent}%")
    if stoploss_pips < 0.5:
        raise ValueError(f"FTMO min SL ist 0.5 Pips: {stoploss_pips}")
    if pip_value <= 0:
        raise ValueError(f"Pip-Wert muss positiv sein: {pip_value}")

    # Risikoberechnung
    risk_amount = balance * (risk_percent / 100)
    raw_lot_size = risk_amount / (stoploss_pips * pip_value)
    
    # FTMO-Limits anwenden
    lot_size = min(round(raw_lot_size, 2), 50.0)  # Hard Cap bei 50 Lots
    
    # Symbol-spezifische Checks (z.B. für Indizes)
    if symbol and "US30" in symbol.upper():
        lot_size = min(lot_size, 5.0)  # Sonderlimit für US30
    
    # Protokollierung für Compliance
    logging.info(
        f"FTMO_LOT_CALC|{symbol or 'GENERIC'}|"
        f"BAL:{balance:.2f}|RISK:{risk_percent}%|"
        f"SL:{stoploss_pips}pips|LOT:{lot_size}"
    )
    
    return lot_size
