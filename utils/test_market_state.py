#!/usr/bin/env python3
# /opt/coreflow/utils/test_market_state.py

import pandas as pd
from core.ki.market_state import MarketStateClassifier

# Dummy-Datenframe mit minimalen Testdaten
df = pd.DataFrame({
    'close': [1.101, 1.102, 1.103, 1.104, 1.105]*10,
    'volume': [100, 150, 130, 120, 140]*10,
    'high': [1.105, 1.106, 1.107, 1.108, 1.109]*10,
    'low': [1.100, 1.101, 1.102, 1.103, 1.104]*10,
})

# Klassifikator instanziieren
classifier = MarketStateClassifier(production_mode=False)

# Marktklassifikation ausgeben
state = classifier.classify(df)
print("📊 Klassifizierter Marktstatus:")
print(state)
