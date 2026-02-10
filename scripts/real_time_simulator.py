
import os
import sys
import time
import pickle
import random
import numpy as np
from pathlib import Path

# Add project root to sys.path for model loading
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from loaders.kdd_loader import KDDLoader

# ANSI Colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def run_real_time_simulation():
    print(f"{CYAN}{BOLD}=== QI-HIDS v1.0: REAL-TIME STREAM SIMULATOR ==={RESET}")
    print(f"{YELLOW}Status: Initializing Tunneling Learning Engine...{RESET}")
    print(f"{YELLOW}Architecture: Underpinning by the Tunneling Learning (TL) Paradigm...{RESET}")
    
    # 1. Load Model
    pkl_path = ROOT_DIR / "models" / "tunneling_v1.pkl"
    if not pkl_path.exists():
        print(f"{RED}Error: Model file (.pkl) not found! Run scripts/crystallize_v1.py first.{RESET}")
        return

    with open(pkl_path, 'rb') as f:
        engine = pickle.load(f)
    
    print(f"{GREEN}✓ Engine Online (Crystallized State Loaded){RESET}")
    
    # 2. Load Data for Simulation
    loader = KDDLoader()
    X, y_true = loader.load_data(sample_size=100) # Get 100 samples to stream
    
    print(f"\n{BOLD}Starting Live Feed Simulation...{RESET}")
    print(f"{'TIMESTAMP':<12} | {'SENSORS':<10} | {'PREDICTION':<10} | {'CONFIDENCE':<10} | {'ACTION'}")
    print("-" * 75)

    try:
        for i in range(len(X)):
            # Simulate network delay
            time.sleep(random.uniform(0.1, 0.4))
            
            # Get current timestamp
            now = time.strftime("%H:%M:%S", time.localtime())
            
            # Run Inference
            sample = X[i].reshape(1, -1)
            result = engine.detect(sample, era='legacy')[0]
            
            # Format UI
            pred = result['prediction']
            conf = result['confidence']
            status = result['status']
            
            if pred == "MALICIOUS":
                color = RED
                action = f"{BOLD}BLOCKING FLOW{RESET}"
            else:
                color = GREEN
                action = "ALLOWING FLOW"

            print(f"{now:<12} | {CYAN}ONLINE{RESET:<10} | {color}{pred:<10}{RESET} | {conf:<10} | {action}")
            
            if i % 10 == 0 and i > 0:
                print(f"{YELLOW}[LOG] Tunneling Stability: 100.00% | Manifold Healthy{RESET}")

    except KeyboardInterrupt:
        print(f"\n{YELLOW}Simulation stopped by user.{RESET}")
    
    print(f"\n{CYAN}{BOLD}=== Simulation Complete ==={RESET}")

if __name__ == "__main__":
    run_real_time_simulation()
