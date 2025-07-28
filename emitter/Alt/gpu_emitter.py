import torch
import redis

class GPUEmitter:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.redis = redis.Redis(
            host="localhost",
            port=6379,
            password=os.getenv("REDIS_PASS")
        )

    def _load_model(self):
        model = DummyTradingModel().to(self.device)
        model.eval()
        return model

    def run(self):
        while True:
            inputs = torch.randn(1, 12).to(self.device)  # Dummy-Daten auf GPU
            with torch.no_grad():
                signal = torch.argmax(self.model(inputs)).item()
            self.redis.publish("gpu_signals", str(signal))
