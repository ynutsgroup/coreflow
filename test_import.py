import importlib

ftmo_risk_manager = importlib.import_module('core.risk_management.ftmo_risk_manager')
FTMORiskManager = ftmo_risk_manager.FTMORiskManager

print("FTMORiskManager imported successfully!")
