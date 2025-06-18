import logging
from datetime import datetime
from typing import Optional, Dict, Any

import torch
import pandas as pd
import numpy as np

# Logger für dieses Modul einrichten (Best Practice: __name__ verwenden)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Keine eigenen Handler hinzufügen:contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9}

class MarketStateClassifier:
    """Klassifiziert den Marktstatus (Trend/Volatilität) anhand von OHLCV-Daten."""
    def __init__(self, model_path: str = "/opt/coreflow/models/market_state_classifier.pt",
                 dummy: bool = False, debug: bool = False):
        """
        Initialisiert den Marktstatus-Klassifizierer.
        Im Produktionsmodus wird ein TorchScript-Modell geladen, im Dummy-Modus erfolgen zufällige Ausgaben.
        """
        self.model_path = model_path
        self.dummy_mode = dummy
        self.debug = debug
        self.model: Optional[torch.jit.ScriptModule] = None
        self.state_labels: list[str] = [
            "Trending Up", "Trending Down", "Ranging", "High Volatility", "Low Volatility"
        ]
        self.last_state: Optional[Dict[str, Any]] = None

        if not self.dummy_mode:
            # Versuche, das TorchScript-Modell zu laden
            try:
                self.model = torch.jit.load(self.model_path)
                self.model.eval()  # Modell in Evaluationsmodus schalten
                logger.info("TorchScript-Modell erfolgreich geladen von %s", self.model_path)
            except Exception as e:
                logger.error("Fehler beim Laden des Modells: %s", e)
                logger.warning("Wechsle in Dummy-Modus (verwende Zufallsdaten statt Modell).")
                self.model = None
                self.dummy_mode = True
        else:
            logger.info("MarketStateClassifier im Dummy-Modus gestartet (kein echtes Modell geladen).")

    def _preprocess(self, df: pd.DataFrame) -> torch.Tensor:
        """Validiert und verarbeitet den DataFrame zu einem Eingabetensor für das Modell."""
        # Sicherstellen, dass df ein DataFrame mit Inhalt ist
        if df is None or df.empty:
            logger.error("Eingabedaten sind leer oder None.")
            raise ValueError("Input DataFrame is empty or None.")
        # Benötigte Spalten (OHLCV)
        required_cols = {"open", "high", "low", "close", "volume"}
        df_cols_lower = {col.lower() for col in df.columns}
        if not required_cols.issubset(df_cols_lower):
            logger.error("DataFrame fehlt mindestens eine der erforderlichen Spalten: %s", required_cols)
            raise ValueError("Input DataFrame must contain OHLCV columns.")
        # Optional: DataFrame nach Index sortieren, falls Index datetime ist
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
        # Filter nur auf benötigte Spalten (im Originalcase)
        # Finde tatsächliche Spaltennamen unabhängig von Groß/Kleinschreibung
        col_map = {col.lower(): col for col in df.columns}
        ohlcv_cols = [col_map[col] for col in ["open", "high", "low", "close", "volume"]]
        data = df[ohlcv_cols].values
        # Überprüfen auf numerische Werte und NaNs
        if not np.issubdtype(data.dtype, np.number):
            logger.error("Die OHLCV-Daten enthalten nicht-numerische Werte.")
            raise ValueError("OHLCV data must be numeric.")
        if np.isnan(data).any():
            logger.error("Die OHLCV-Daten enthalten ungültige NaN-Werte.")
            raise ValueError("OHLCV data contains NaN values.")
        # Konvertieren zu float32 numpy (für torch Kompatibilität)
        data = data.astype(np.float32)
        # In Torch-Tensor umwandeln
        tensor = torch.from_numpy(data)
        # Batch-Dimension hinzufügen, falls Sequenzdaten erwartet
        tensor = tensor.unsqueeze(0)  # shape: (1, seq_length, features)
        return tensor

    def classify(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Klassifiziert den Marktstatus basierend auf den Eingabedaten (DataFrame)."""
        # Verarbeite Eingabedaten (Validierung und Feature-Extraktion)
        try:
            features_tensor = self._preprocess(df)
        except Exception as e:
            # Fehler bereits in _preprocess() geloggt
            raise

        # Bestimme Timestamp für Ausgabe (letzter Index oder aktuelle Zeit)
        timestamp: datetime
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
            timestamp = df.index[-1]
        elif "timestamp" in df.columns and len(df["timestamp"]) > 0:
            timestamp = df["timestamp"].iloc[-1]
        else:
            timestamp = datetime.utcnow()
        # In ISO-Format (UTC) umwandeln, falls datetime-Objekt
        if isinstance(timestamp, datetime):
            # Sicherstellen, dass wir einen UTC-Zeitstempel als String erhalten
            try:
                timestamp_str = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                timestamp_str = str(timestamp)
        else:
            timestamp_str = str(timestamp)

        result: Dict[str, Any] = {
            "market_state": None,
            "confidence": None,
            "timestamp": timestamp_str,
            "description": None
        }

        if self.dummy_mode:
            # Dummy-Modus: Generiere Zufallszustand
            import random
            state = random.choice(self.state_labels)
            # Setze Confidence auf einen Zufallswert zwischen 0.5 und 1.0 (zwei Nachkommastellen)
            confidence = round(random.uniform(0.5, 1.0), 2)
            # Beschreibung basierend auf dem gewählten Status
            descriptions = {
                "Trending Up": "Market trending upward with strong momentum.",
                "Trending Down": "Market trending downward with selling pressure.",
                "Ranging": "Market moving sideways in a range-bound fashion.",
                "High Volatility": "Market showing high volatility and large swings.",
                "Low Volatility": "Market is calm with low volatility."
            }
            description = descriptions.get(state, "No description available.")
            result.update({
                "market_state": state,
                "confidence": confidence,
                "description": description
            })
            logger.debug("Dummy-Ausgabe: %s", result)
        else:
            # Produktionsmodus: Modell-Inferenz durchführen
            try:
                with torch.no_grad():  # Keine Gradientenberechnung während Inferenz:contentReference[oaicite:10]{index=10}
                    model_out = self.model(features_tensor)
            except Exception as e:
                logger.error("Modell-Inferenz schlug fehl: %s", e)
                raise

            # Verarbeitung des Modeloutputs zu Label + Confidence
            try:
                # Erwartet: model_out als Tensor (1 x num_classes) oder (num_classes,)
                output_tensor = model_out
                # Falls der Output ein Tensor mit mehr als 1 Element ist, Softmax für Wahrscheinlichkeiten
                if output_tensor.dim() == 0 or output_tensor.numel() == 1:
                    # Spezialfall: Modell gibt direkt einen einzelnen Wert (z.B. Klassenindex) aus
                    pred_idx = int(output_tensor.item())
                    probabilities = None
                else:
                    if output_tensor.dim() > 1:
                        # Reduziere auf 2D: (batch, classes)
                        output_tensor = output_tensor.view(1, -1) if output_tensor.dim() == 1 else output_tensor
                        # Nehme ggf. nur erstes Batch-Element
                        output_tensor = output_tensor[0]
                    # Berechne Softmax-Wahrscheinlichkeiten über Klassen
                    probabilities = torch.softmax(output_tensor, dim=0).cpu().numpy()
                    pred_idx = int(np.argmax(probabilities))
                # Map Index zu Klassenname
                if 0 <= pred_idx < len(self.state_labels):
                    state = self.state_labels[pred_idx]
                else:
                    state = "Unknown"
                # Confidence bestimmen
                if probabilities is not None:
                    confidence = float(probabilities[pred_idx])
                else:
                    # Wenn keine Wahrscheinlichkeiten berechnet (z.B. direkter Index), Dummy-Confidence 1.0
                    confidence = 1.0
                # Beschreibung wie im Dummy aus Dictionary holen (oder generisch, falls Unknown)
                descriptions = {
                    "Trending Up": "Market trending upward with strong momentum.",
                    "Trending Down": "Market trending downward with selling pressure.",
                    "Ranging": "Market moving sideways in a range-bound fashion.",
                    "High Volatility": "Market showing high volatility and large swings.",
                    "Low Volatility": "Market is calm with low volatility."
                }
                description = descriptions.get(state, "No description available.")
                result.update({
                    "market_state": state,
                    "confidence": confidence,
                    "description": description
                })
                logger.info("Klassifizierung durchgeführt: Zustand=%s, Confidence=%.2f", state, confidence)
            except Exception as e:
                logger.error("Fehler bei der Verarbeitung des Model-Outputs: %s", e)
                raise

        # Last State Cache aktualisieren
        self.last_state = result
        # Bei Debug: Features dem Resultat beifügen (zur Einsicht)
        if self.debug:
            # features_tensor zu Python-Objekt konvertieren, z.B. Liste oder Nested List
            try:
                features_data = features_tensor.cpu().numpy().tolist()
            except Exception:
                features_data = None
            result["features"] = features_data
            logger.debug("Features (Debug-Modus) dem Result hinzugefügt.")
        return result
