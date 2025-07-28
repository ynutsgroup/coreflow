#!/usr/bin/env python3
# CoreFlow Institutional AI Emitter - Redis Signal Dispatcher

import os
import time
import json
import torch
import redis
import logging
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
from torch import nn

# === ENV & Logging ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env.ai_emitter_pro"))

logger = logging.getLogger("CF.Emitter.CPU")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s")
console = logging.StreamHandler()
file = logging.FileHandler(os.path.join(os.getenv("LOG_DIR", "/tmp"), "ai_emitter_pro.log"))
console.setFormatter(formatter)
file.setFormatter(formatter)
logger.addHandler(console)
logger.addHandler(file)

# === Dummy-Modell (kompatibel mit AI_INPUT_SIZE) ===
class DummyTradingModel(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, output_size=3):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return torch.softmax(self.layer2(x), dim=1)

# === Modell Initialisierung ===
def initialize_model():
    path = os.getenv("AI_MODEL_PATH")
    device = torch.device("cuda" if torch.cuda.is_available() and os.getenv("TRADE_MODE") == "LIVE" else "cpu")
    model = DummyTradingModel(
        input_size=int(os.getenv("AI_INPUT_SIZE")),
        hidden_size=int(os.getenv("AI_HIDDEN_SIZE")),
        output_size=int(os.getenv("AI_OUTPUT_SIZE"))
    ).to(device)

    try:
        state = torch.load(path, map_location=device)
        model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
        logger.info(f"Model loaded from {path}")
    except Exception as e:
        logger.error(f"Fehler beim Laden des Modells: {e}")
        raise

    model.eval()
    return model, device

# === Dummy-Eingabe erzeugen ===
def generate_dummy_data(length=12):
    return np.random.rand(1, length).astype(np.float32)

# === Trade-Signal senden ===
def send_trade_signal(pred: int):
    redis_channel = os.getenv("REDIS_CHANNEL")
    symbol = os.getenv("TEST_SYMBOL")
    lot = float(os.getenv("TEST_VOLUME"))
    sl = int(os.getenv("TRADE_SL"))
    tp = int(os.getenv("TRADE_TP"))
    action = "BUY" if pred == 1 else "SELL" if pred == 2 else "WAIT"

    if action == "WAIT":
        logger.info("→ WAIT: Kein Trade gesendet.")
        return

    signal = {
        "symbol": symbol,
        "action": action,
        "lot": lot,
        "sl": sl,
        "tp": tp,
        "timestamp": datetime.utcnow().isoformat()
    }

    try:
        r = redis.Redis(
            host=os.getenv("REDIS_HOST"),
            port=int(os.getenv("REDIS_PORT")),
            password=os.getenv("REDIS_PASSWORD"),
            decode_responses=True
        )
        r.publish(redis_channel, json.dumps(signal))
        logger.info(f"Signal gesendet: {signal}")
    except Exception as e:
        logger.error(f"Redis-Fehler: {e}")

# === Hauptfunktion ===
def main():
    logger.info("=== Starting AI Emitter ===")
    model, device = initialize_model()
    input_size = int(os.getenv("AI_INPUT_SIZE"))

    while True:
        try:
            dummy_input = generate_dummy_data(input_size)
            tensor = torch.tensor(dummy_input).to(device)
            prediction = torch.argmax(model(tensor), dim=1).item()
            send_trade_signal(prediction)
        except Exception as e:
            logger.error(f"Fehler im Loop: {e}")
        time.sleep(float(os.getenv("EMIT_INTERVAL", 10)))

if __name__ == "__main__":
    main()
