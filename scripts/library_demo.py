
"""
QI-HIDS Library Demo: How to use UniversalTunnelingNetwork
This script demonstrates how to dynamically add new dataset gates (IoT, 5G, etc.)
to the core Tunneling Learning engine without modifying source code.
"""

import torch
import sys
from pathlib import Path

# Setup Python Path to include root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from models.tunneling_lib import UniversalTunnelingNetwork

def demo_library_usage():
    print("🚀 QI-HIDS Universal Tunneling Library Demo\n")
    
    # ---------------------------------------------------------
    # STEP 1: Initialize the Universal Engine
    # ---------------------------------------------------------
    # The 'latent_dim' is the size of the shared Tunneling Manifold.
    # This "Brain" is shared across all datasets.
    model = UniversalTunnelingNetwork(latent_dim=128)
    print("✓ [Core] Engine Initialized (Shared Manifold Online)")

    # ---------------------------------------------------------
    # STEP 2: Add New Datasets (Gates) Dynamically
    # ---------------------------------------------------------
    # You can register ANY data source with ANY feature count.
    
    # Example: Smart Home IoT Sensors (Low dimensionality)
    print("→ [Gate] Adding 'Smart-Home-IoT' gate (Feature Dim: 12)...")
    model.add_dataset(name='iot_home', input_dim=12)
    
    # Example: 5G Network Slicing (High dimensionality)
    print("→ [Gate] Adding '5G-Core-Slice' gate (Feature Dim: 150)...")
    model.add_dataset(name='5g_core', input_dim=150)
    
    # ---------------------------------------------------------
    # STEP 3: Run Detection (Route by Source) WITH NORMALIZATION
    # ---------------------------------------------------------
    print("\n🔍 Running Inference Simulation:")
    
    # Case A: IoT Traffic (Raw Values)
    dummy_iot = torch.randn(10, 12) * 500 # Unscaled raw data
    
    # ⚡ NEW: Apply built-in Normalization
    dummy_iot_norm = UniversalTunnelingNetwork.normalize(dummy_iot)
    
    # KEY API CALL: source='iot_home'
    output_iot = model(dummy_iot_norm, source='iot_home')
    
    print(f"  Detected IoT Flow (Source='iot_home'): {output_iot.detach().numpy()[0]}")

    print("\n✅ Library Usage Verified. The system is ready for new datasets.")

if __name__ == "__main__":
    demo_library_usage()
