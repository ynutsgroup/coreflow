import os
import numpy as np
import onnxruntime as ort

def generate_signal(symbol: str) -> str:
    """
    Führt ONNX-Modell-Inferenz für das gegebene Symbol durch.
    Erwartet Modell unter /opt/coreflow/models/{symbol}.onnx
    """

    model_path = f"/opt/coreflow/models/{symbol.lower()}.onnx"
    if not os.path.exists(model_path):
        print(f"[ONNX] Kein Modell für {symbol}, fallback → HOLD")
        return "HOLD"

    try:
        sess_options = ort.SessionOptions()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(model_path, sess_options, providers=providers)

        # Dummy-Eingabedaten (z. B. technischer Indikatoren)
        dummy_input = np.random.rand(1, 64).astype(np.float32)
        input_name = session.get_inputs()[0].name

        output = session.run(None, {input_name: dummy_input})[0]
        probs = output[0]  # z. B. [0.1, 0.8, 0.1]

        confidence = float(np.max(probs))
        action_idx = int(np.argmax(probs))
        action_map = {0: "BUY", 1: "SELL", 2: "HOLD"}
        action = action_map.get(action_idx, "HOLD")

        print(f"[ONNX] {symbol} → {action} ({confidence*100:.1f}%)")
        return action if confidence >= 0.7 else "HOLD"
    except Exception as e:
        print(f"[ONNX] Fehler bei Inferenz für {symbol}: {e}")
        return "HOLD"
