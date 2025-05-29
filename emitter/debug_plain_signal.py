import redis

# Redis Verbindung
r = redis.Redis(
    host="127.0.0.1",
    port=6379,
    password="NeuesSicheresPasswort#2024!"
)

# Unverschlüsseltes Test-Signal senden
r.xadd("trading_signals", {
    "symbol": "EURUSD",
    "action": "BUY"
})

print("✅ Test-Signal ohne Verschlüsselung gesendet.")
