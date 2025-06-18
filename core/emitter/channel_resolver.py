# /opt/coreflow/core/emitter/channel_resolver.py

import yaml
import os

ROUTING_PATH = os.getenv("ROUTING_CONFIG", "/opt/coreflow/config/routing.yaml")

def resolve_channel(symbol: str) -> str:
    try:
        with open(ROUTING_PATH, 'r') as f:
            config = yaml.safe_load(f)
        return config["channels"].get(symbol.upper(), config["channels"].get("DEFAULT", "trading_signals"))
    except Exception as e:
        raise RuntimeError(f"Kanalauflösung fehlgeschlagen: {e}")
