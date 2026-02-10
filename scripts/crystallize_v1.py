"""
QI-HIDS v1.0: MODEL CRYSTALLIZER
Transforms the raw .pth weights and Python logic into a single 'Crystallized' .pkl object.
This file can be loaded in production with 1 line of code to provide high-confidence IDS.
"""

import pickle
import torch
import sys
from pathlib import Path

# Ensure root is in path to find models/v1_engine
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from models.tunneling_v1_api import TunnelingLearningInference

def crystallize():
    print("💎 Crystallizing QI-HIDS v1 Manifold...")
    
    # Initialize the engine (this loads the .pth internally)
    # We use the dimensions detected during last training
    k_dim = 36
    c_dim = 78
    
    try:
        engine = TunnelingLearningInference(
            pth_path="models/tunneling_v1.pth",
            k_dim=k_dim,
            c_dim=c_dim
        )
        
        # Save the entire instance
        save_path = ROOT_DIR / "models" / "tunneling_v1.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(engine, f)
            
        print(f"✓ CRYSTALLIZATION COMPLETE: {save_path}")
        print("  The .pkl now contains Architecture + Intelligence + Weights.")
        
    except Exception as e:
        print(f"FAILED to crystallize: {e}")

if __name__ == "__main__":
    crystallize()
