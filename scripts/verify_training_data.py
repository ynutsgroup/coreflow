import numpy as np
import os

X_PATH = "/opt/coreflow/data/training/features/X.npy"
Y_PATH = "/opt/coreflow/data/training/labels/y.npy"

# Laden
X = np.load(X_PATH)
y = np.load(Y_PATH)

print("✅ Dateien geladen")
print(f"🔢 X Shape (original): {X.shape}")
print(f"🔢 y Shape: {y.shape}")

# Prüfen auf 4D → squeeze
if X.ndim == 4:
    print("⚠️ X ist 4D, squeeze wird angewendet.")
    X = X.squeeze(1)
    print(f"✅ Neues X Shape: {X.shape}")

# Prüfung auf Dimensionen
if X.ndim != 3:
    raise ValueError("❌ X muss 3D sein (batch, seq_len, input_size)")

if len(y.shape) != 1:
    raise ValueError("❌ y muss 1D sein (Klassenzuordnung)")

if X.shape[0] != y.shape[0]:
    raise ValueError(f"❌ Anzahl der Samples ungleich: X={X.shape[0]}, y={y.shape[0]}")

# Label-Diagnose
classes, counts = np.unique(y, return_counts=True)
print("🧾 Klassenverteilung:")
for cls, cnt in zip(classes, counts):
    print(f" - Klasse {cls}: {cnt} Samples")

# Optional abspeichern der bereinigten Daten
np.save("/opt/coreflow/data/training/features/X_clean.npy", X)
print("✅ Bereinigte X gespeichert als X_clean.npy")
