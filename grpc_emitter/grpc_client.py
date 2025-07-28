import grpc
import trading_pb2
import trading_pb2_grpc
import os
import sys
from dotenv import load_dotenv

load_dotenv('/opt/coreflow/config/.env.grpc')

def run(symbol: str):
    host = os.getenv("GRPC_SERVER_HOST")
    port = os.getenv("GRPC_SERVER_PORT")
    if not host or not port:
        raise ValueError("gRPC-Host oder Port fehlt in .env.grpc")

    channel = grpc.insecure_channel(f'{host}:{port}')
    stub = trading_pb2_grpc.TradingServiceStub(channel)

    response = stub.SendSignal(trading_pb2.SignalRequest(symbol=symbol))
    print(f"[gRPC Client] Signal empfangen: {response.action} für {symbol}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Aufruf: python3 grpc_client.py SYMBOL")
        sys.exit(1)
    run(sys.argv[1])
