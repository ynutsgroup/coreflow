# Sicherheitsschleife für Testbetrieb
import sys
confirm = input("🚨 Dieser Emitter wird alle 30 Sekunden Signale senden. Fortfahren? (yes/no): ")
if confirm.strip().lower() != 'yes':
    print("❌ Abgebrochen.")
    sys.exit(0)
