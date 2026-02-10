import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Setup Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from models.v1_engine import build_v1
from loaders.kdd_loader import KDDLoader
from loaders.cicids2017_loader import CICIDS2017Loader

def main():
    print("QI-HIDS v1.0: PEAK STRESS CHAOS GAUNTLET")
    print("---------------------------------------")
    
    device = torch.device('cpu')
    
    try:
        cic = CICIDS2017Loader()
        kdd = KDDLoader()
        X_k, _ = kdd.load_data(sample_size=10)
        X_c, _ = cic.load_data(sample_size=100)
        k_dim, c_dim = X_k.shape[1], X_c.shape[1]
        print(f"Dimensions: K={k_dim}, C={c_dim}")
        
        # Load heavy batch
        X_test, y_test = cic.load_data(sample_size=500)
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_tensor = torch.tensor(y_test, dtype=torch.long)
    except Exception as e:
        print(f"Data Init Error: {e}")
        return

    model = build_v1(legacy_dim=k_dim, modern_dim=c_dim)
    model.load_state_dict(torch.load(ROOT_DIR / "models" / "tunneling_v1.pth", map_location=device))
    model.eval()
    
    print("\n--- INITIATING PERFECT STORM ATTACK ---")
    
    # 1. Adversarial FGSM (Eps=0.15)
    X_adv = X_tensor.clone().detach()
    X_adv.requires_grad = True
    out = model(X_adv, era='modern')
    loss = nn.NLLLoss()(torch.log(out + 1e-10), y_tensor)
    loss.backward()
    X_chaos = X_tensor + 0.15 * X_adv.grad.data.sign()
    
    # 2. 80% Feature Loss
    mask = torch.rand_like(X_chaos) > 0.8
    X_chaos *= mask
    
    # 3. High Jitter (Sigma=0.4)
    X_chaos += torch.randn_like(X_chaos) * 0.4
    
    # 4. Segment Blackout (First 30 features)
    X_chaos[:, :30] = 0
    
    with torch.no_grad():
        final_out = model(X_chaos, era='modern')
        preds = torch.argmax(final_out, dim=1)
        
        acc = (preds == y_tensor).float().mean().item()
        conf = final_out.max(dim=1).values.mean().item()
        
    print(f"\nPEAK STRESS RESULTS (500 FLOWS):")
    print(f"  Absolute Accuracy: {acc:.2%}")
    print(f"  Decision Confidence: {conf:.2%}")
    
    if acc > 0.95:
        print("\n🏆 VERDICT: THE MANIFOLD IS INDESTRUCTIBLE.")
    else:
        print("\n⚠️ VERDICT: SYSTEM DEGRADED.")

if __name__ == "__main__":
    main()
