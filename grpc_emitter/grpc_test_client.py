import grpc
import trading_pb2
import trading_pb2_grpc

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = trading_pb2_grpc.TradingServiceStub(channel)

    response = stub.SendSignal(trading_pb2.SignalRequest(symbol="EURUSD"))
    print(f"Empfangenes Signal: {response.action}")

if __name__ == "__main__":
    run()
