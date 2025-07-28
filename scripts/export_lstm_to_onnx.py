"""
Institutional-Grade LSTM Model Export for Trading
FTMO-Compliant Implementation

Features:
- Complete audit trail
- Checksum verification
- Production-grade error handling
- Model validation
- Compliance logging
"""

import torch
import torch.nn as nn
import yaml
import os
import hashlib
import json
from datetime import datetime
import onnx
import onnxruntime as ort

# 🔐 Secure Path Configuration
MODEL_DIR = "/opt/coreflow/models/lstm"
CONFIG_DIR = "/opt/coreflow/config/models"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.pth")
ONNX_PATH = os.path.join(MODEL_DIR, "lstm_model.onnx")
CONFIG_PATH = os.path.join(CONFIG_DIR, "lstm_config.yaml")
AUDIT_LOG = os.path.join(MODEL_DIR, "audit_log.json")

class FTMOCompliance:
    """FTMO compliance and validation utilities"""
    
    @staticmethod
    def generate_checksum(file_path):
        """Generate SHA-256 checksum for model verification"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    @staticmethod
    def log_audit_entry(action, status, metadata=None):
        """Maintain audit trail for compliance"""
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "action": action,
            "status": status,
            "metadata": metadata or {}
        }
        
        # Read-modify-write audit log
        try:
            if os.path.exists(AUDIT_LOG):
                with open(AUDIT_LOG, 'r') as f:
                    log = json.load(f)
            else:
                log = []
                
            log.append(entry)
            
            with open(AUDIT_LOG, 'w') as f:
                json.dump(log, f, indent=2)
        except Exception as e:
            print(f"⚠️ Audit logging failed: {str(e)}")

class TradingLSTM(nn.Module):
    """Institutional-Grade LSTM Model"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

def export_model():
    """Main export procedure with compliance checks"""
    try:
        # Initialize compliance logging
        FTMOCompliance.log_audit_entry("export_initiated", "started")
        
        # 1. Load and verify configuration
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
            
        required_params = ['input_size', 'hidden_size', 'num_layers', 
                         'output_size', 'seq_length']
        if not all(param in config for param in required_params):
            raise ValueError("Invalid configuration: missing required parameters")
            
        # 2. Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TradingLSTM(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=config['output_size']
        ).to(device)
        
        # 3. Load weights with verification
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            
        original_checksum = FTMOCompliance.generate_checksum(MODEL_PATH)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        
        # 4. Prepare for export
        dummy_input = torch.randn(1, config['seq_length'], config['input_size']).to(device)
        
        # 5. Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            ONNX_PATH,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            },
            verbose=False
        )
        
        # 6. Post-export validation
        onnx_model = onnx.load(ONNX_PATH)
        onnx.checker.check_model(onnx_model)
        
        # Verify ONNX model can be loaded
        ort_session = ort.InferenceSession(ONNX_PATH)
        ort_inputs = {ort_session.get_inputs()[0].name: 
                     dummy_input.cpu().numpy()}
        ort_session.run(None, ort_inputs)
        
        # 7. Final checksum and logging
        exported_checksum = FTMOCompliance.generate_checksum(ONNX_PATH)
        metadata = {
            "original_checksum": original_checksum,
            "exported_checksum": exported_checksum,
            "config": config,
            "environment": {
                "pytorch_version": torch.__version__,
                "onnx_version": onnx.__version__,
                "device": str(device)
            }
        }
        
        FTMOCompliance.log_audit_entry("export_completed", "success", metadata)
        print(f"✅ Institutional-grade export completed\n"
              f"   - Model: {ONNX_PATH}\n"
              f"   - Checksum: {exported_checksum}\n"
              f"   - Audit log: {AUDIT_LOG}")
        
    except Exception as e:
        FTMOCompliance.log_audit_entry("export_failed", "failed", {"error": str(e)})
        print(f"❌ Export failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    export_model()
