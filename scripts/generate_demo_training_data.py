import numpy as np
import os

# Parameter
samples = 1000
seq_len = 60
input_size = 10
num_classes = 3

# Erzeugung
X = np.random.randn(samples, seq_len, input_size).astype(np.float32)
y = np.random.randint(0, num_classes, size=(samples,)).astype(np.int64)

# Verzeichnisse
os.makedirs("/opt/coreflow/data/training/features", exist_ok=True)
os.makedirs("/opt/coreflow/data/training/labels", exist_ok=True)

# Speicherung
np.save("/opt/coreflow/data/training/features/X_demo.npy", X)
np.save("/opt/coreflow/data/training/labels/y_demo.npy", y)

print("✅ X_demo.npy und y_demo.npy erzeugt.")
