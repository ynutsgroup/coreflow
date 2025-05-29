#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import pandas as pd
from ta import add_all_ta_features

# GPU-Konfiguration
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# LSTM-Modell
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(60, 10)),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Datenverarbeitung
def preprocess_data(mt5_data):
    df = pd.DataFrame(mt5_data)
    df = add_all_ta_features(df, open="open", high="high",
                           low="low", close="close", volume="tick_volume")
    return df.dropna()

def generate_signal(live_data):
    processed = preprocess_data(live_data)
    X = np.array([processed[-60:]])  # Letzte 60 Kerzen
    prediction = model.predict(X)
    return ['BUY', 'HOLD', 'SELL'][np.argmax(prediction)]
