class TradeEmitter:
    def __init__(self):
        self.redis = redis.Redis(host="localhost", port=6379)
        self.pubsub = self.redis.pubsub()
        self.pubsub.subscribe("gpu_signals")

    def process(self):
        for message in self.pubsub.listen():
            if message["type"] == "message":
                signal = int(message["data"])
                # Hier: FTMO-Risikoprüfung + MT5-Sendelogik
                print(f"Signal empfangen: {signal}")
