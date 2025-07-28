import grpc
import trading_pb2
import trading_pb2_grpc
import os
import sys
from dotenv import load_dotenv

env_loaded = load_dotenv('/opt/coreflow/config/.env.grpc')
print(f"[DEBUG] .env geladen: {env_loaded}")

host = os.getenv("GRPC_SERVER_HOST")
port = os.getenv("GRPC_SERVER_PORT")
print(f"[DEBUG] Zieladresse: {host}:{port}")

if not host or not port:
    print("[ERROR] .env.grpc nicht gefunden oder leer")
    sys.exit(1)

try:
    print("[DEBUG] Baue Verbindung auf...")
    channel = grpc.insecure_channel(f'{host}:{port}')
    grpc.channel_ready_future(channel).result(timeout=3)
    print("[DEBUG] Verbindung zu gRPC-Server steht.")
except Exception as e:
    print(f"[ERROR] Verbindung fehlgeschlagen: {e}")
    sys.exit(1)

stub = trading_pb2_grpc.TradingServiceStub(channel)

symbol = "EURUSD"
print(f"[DEBUG] Sende Anfrage für: {symbol}")

try:
    response = stub.SendSignal(trading_pb2.SignalRequest(symbol=symbol))
    print(f"[OK] Empfangen: {response.action}")
except Exception as e:
    print(f"[ERROR] RPC fehlgeschlagen: {e}")
