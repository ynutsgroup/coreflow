# /opt/coreflow/core/utils/env_validator.py

import os
import yaml
from dotenv import load_dotenv

load_dotenv('/opt/coreflow/.env')

ROUTING_CONFIG = os.getenv("ROUTING_CONFIG", "/opt/coreflow/config/routing.yaml")
explicit_channel = os.getenv("REDIS_CHANNEL")

try:
    with open(ROUTING_CONFIG, 'r') as f:
        routing = yaml.safe_load(f)
except Exception as e:
    print(f"❌ Routing-Konfig nicht lesbar: {e}")
    routing = {"channels": {}}

symbols = routing.get("channels", {}).keys()
def_channel = routing.get("channels", {}).get("DEFAULT", "trading_signals")

print("🔍 ENV-Validierung:")

if explicit_channel:
    print(f"✅ Fester Redis-Kanal gesetzt: {explicit_channel}")
    if explicit_channel not in routing["channels"].values():
        print("⚠️  WARNUNG: Kanal nicht in routing.yaml definiert")
else:
    print("ℹ️ Kein REDIS_CHANNEL in .env → Symbolbasiertes Routing aktiv")
    print(f"📚 Bekannte Kanäle laut YAML: {', '.join(symbols)}")
    print(f"🛡️ Fallback-Kanal: {def_channel}")
