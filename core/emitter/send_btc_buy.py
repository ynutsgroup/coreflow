#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Sendet ein verschlüsseltes BUY-Signal für BTCUSD via Redis

from signal_emitter import send_signal

if __name__ == "__main__":
    try:
        send_signal(
            symbol="BTCUSD",
            action="BUY",
            confidence=0.88,
            volume=0.02
        )
    except Exception as e:
        print(f"Fehler beim Senden: {e}")
