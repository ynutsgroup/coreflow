import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
import numpy as np

# 📄 Konfiguration
X_PATH = "/opt/coreflow/data/training/features/X_demo.npy"
Y_PATH = "/opt/coreflow/data/training/labels/y_demo.npy"
CONFIG_PATH = "/opt/coreflow/config/models/lstm_config.yaml"

print(f"📥 Lade Konfig aus: {CONFIG_PATH}")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

EPOCHS = config["epochs"]
BATCH_SIZE = config["batch_size"]
INPUT_SIZE = config["input_size"]
HIDDEN_SIZE = config["hidden_size"]
NUM_LAYERS = config["num_layers"]
OUTPUT_SIZE = config["output_size"]
SEQ_LENGTH = config["seq_length"]

print(f"📊 Hyperparameter: EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}")

# 📦 Lade Daten
print(f"📂 Lade: {X_PATH}")
X = np.load(X_PATH)
y = np.load(Y_PATH)

if X.ndim == 4:
    print("⚠️  X ist 4D, wende squeeze(1) an.")
    X = X.squeeze(1)

print("✅ Shapes: ", X.shape, y.shape)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
print(f"📦 Samples im Dataset: {len(dataset)}")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 📊 Modell
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        h0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_SIZE).to(x.device)
        c0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_SIZE).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

print("🚀 Starte Training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"📈 Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

# 💾 Speichern
MODEL_PATH = "/opt/coreflow/models/lstm/lstm_model.pth"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Modell gespeichert: {MODEL_PATH}")
