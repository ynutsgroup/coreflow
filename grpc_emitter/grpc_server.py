import sys
import os
import time
from concurrent import futures

import grpc
from dotenv import load_dotenv

# 🔧 Import-Pfade anpassen
sys.path.append('/opt/coreflow')

# 📦 gRPC-Protokolle
import trading_pb2
import trading_pb2_grpc

# 🔍 Signal-Logik & Risk Manager
from signal_logic import generate_signal
from core.ftmo_risk_manager import calculate_lot_size, is_ftmo_compliant
from core.signal_emitter import emit_signal

# 🌍 .env laden (Pfad: .env.grpc)
load_dotenv('/opt/coreflow/config/.env.grpc')

# gRPC Servicer-Klasse
class TradingServicer(trading_pb2_grpc.TradingServiceServicer):
    def SendSignal(self, request, context):
        symbol = request.symbol
        print(f"[gRPC] Anfrage empfangen für Symbol: {symbol}")

        # 🧠 KI/VALG-Signal ermitteln
        action = generate_signal(symbol)

        # 📊 Lot-Berechnung + FTMO-Konformität prüfen
        lot = calculate_lot_size(symbol)
        compliant = is_ftmo_compliant(symbol)

        # ✅ Nur wenn konform: weiterleiten an Redis/ZMQ
        if compliant:
            emit_signal(symbol, action, lot)
            print(f"[gRPC] Weitergeleitet: {symbol} {action} {lot}")
        else:
            action = "HOLD"
            print(f"[gRPC] NICHT FTMO-Konform → Signal verworfen")

        return trading_pb2.SignalResponse(action=action)

# 🛰️ Server starten
def serve():
    host = os.getenv("GRPC_SERVER_HOST", "127.0.0.1")
    port = os.getenv("GRPC_SERVER_PORT", "51515")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    trading_pb2_grpc.add_TradingServiceServicer_to_server(TradingServicer(), server)
    server.add_insecure_port(f'{host}:{port}')
    server.start()
    print(f"[gRPC Server] Läuft auf {host}:{port}")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)
        print("\n[gRPC Server] Manuell beendet.")

if __name__ == '__main__':
    serve()
