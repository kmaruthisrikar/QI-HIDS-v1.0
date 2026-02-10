import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.v1_engine import build_v1
from loaders.kdd_loader import KDDLoader
from loaders.cicids2017_loader import CICIDS2017Loader

class ChaosAugmentor:
    """
    Applies real-world noise simulations to feature vectors.
    """
    @staticmethod
    def apply(x, p=0.3):
        x_aug = x.clone()
        
        # 1. Feature Dropout (Telemetry Loss) - Aggressive 50% to match Scenario A
        mask = torch.rand_like(x_aug) > 0.5
        x_aug *= mask
        
        # 2. Gaussian Noise (Encryption Jitter) - Increased for robustness
        noise = torch.randn_like(x_aug) * 0.1
        x_aug += noise
        
        # 3. Header Zeroing (Protocol Obfuscation)
        if np.random.rand() < 0.3:
            x_aug[:, :10] = 0
            
        return x_aug

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    # Init Loaders
    kdd_loader = KDDLoader()
    cic_loader = CICIDS2017Loader()
    
    # Load Subsets for Training
    X_kdd, y_kdd = kdd_loader.load_data(sample_size=10000)
    X_cic, y_cic = cic_loader.load_data(sample_size=10000)
    
    k_dim = X_kdd.shape[1]
    c_dim = X_cic.shape[1]
    print(f"Input Dimensions: KDD={k_dim}, CIC={c_dim}")

    # Convert to Tensors
    train_kdd = TensorDataset(torch.tensor(X_kdd, dtype=torch.float32), torch.tensor(y_kdd, dtype=torch.long))
    train_cic = TensorDataset(torch.tensor(X_cic, dtype=torch.float32), torch.tensor(y_cic, dtype=torch.long))
    
    loader_kdd = DataLoader(train_kdd, batch_size=64, shuffle=True)
    loader_cic = DataLoader(train_cic, batch_size=64, shuffle=True)
    
    model = build_v1(legacy_dim=k_dim, modern_dim=c_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Interleave Eras
        for (batch_kdd, batch_cic) in zip(loader_kdd, loader_cic):
            optimizer.zero_grad()
            
            # --- Legacy Pass (KDD) ---
            x_k, y_k = batch_kdd
            x_k = ChaosAugmentor.apply(x_k.to(device))
            out_k = model(x_k, era='legacy')
            loss_k = criterion(out_k, y_k.to(device))
            
            # --- Modern Pass (CIC) ---
            x_c, y_c = batch_cic
            x_c = ChaosAugmentor.apply(x_c.to(device))
            out_c = model(x_c, era='modern')
            loss_c = criterion(out_c, y_c.to(device))
            
            loss = loss_k + loss_c
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader_kdd):.4f}")

    # Save
    torch.save(model.state_dict(), "models/tunneling_v1.pth")
    print("Model saved to models/tunneling_v1.pth")

if __name__ == "__main__":
    train()
