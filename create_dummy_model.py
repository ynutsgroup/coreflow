#!/usr/bin/env python3
# /opt/coreflow/utils/create_valid_dummy_model.py

import torch
import torch.nn as nn
from pathlib import Path

class DummyTradingModel(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, output_size=3):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return torch.softmax(self.layer2(x), dim=1)

if __name__ == "__main__":
    model = DummyTradingModel(input_size=12, hidden_size=64, output_size=3)
    model_path = Path("/opt/coreflow/models/dummy_cpu_model.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # 🔥 Korrekt: Nur state_dict speichern – KEIN dict drumherum!
    torch.save(model.state_dict(), str(model_path))

    print(f"✅ DummyTradingModel korrekt gespeichert unter: {model_path}")
