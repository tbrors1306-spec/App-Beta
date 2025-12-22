import sys
import subprocess
import time

print("--- STARTE INSTALLATION ---")
print("Installiere streamlit-aggrid...")

try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit-aggrid"])
    print("\n✅ ERFOLG! Das Paket wurde installiert.")
    print("Du kannst dieses Fenster schließen und PipeCraft neu starten.")
except Exception as e:
    print(f"\n❌ FEHLER: {e}")

time.sleep(10) # Fenster bleibt 10 Sekunden offen
