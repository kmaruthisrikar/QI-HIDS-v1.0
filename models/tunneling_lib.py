
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumTunnel(nn.Module):
    """
    Stabilizes latent geometry by dampening high-variance shifts using the Tunneling Learning paradigm.
    """
    def __init__(self, channels, tunnel_width=0.15):
        super().__init__()
        self.barrier = nn.Parameter(torch.ones(channels) * tunnel_width)

    def forward(self, x):
        # Apply a soft-clamping effect relative to the barrier
        return torch.tanh(x / (self.barrier + 1e-6)) * self.barrier

class UniversalTunnelingNetwork(nn.Module):
    """
    QI-HIDS Universal Tunneling Learning Library.
    A flexible, extensible architecture for dynamic multi-gate tunneling.
    
    Usage:
        model = UniversalTunnelingNetwork()
        model.add_dataset('iot', 115)
        model.add_dataset('cloud', 45)
        output = model(features, source='iot')
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.projections = nn.ModuleDict()  # Dynamic gates
        
        # Shared Behavioral Manifold (The "Intelligence" Core)
        self.manifold = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
        
        # Quantum Stabilization Layer
        self.tunnel = QuantumTunnel(latent_dim, tunnel_width=0.15)
        
        # Smooth Decision Head
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 2) # Binary: [Normal, Malicious]
        )
    
    def add_dataset(self, name, input_dim):
        """Add a new projection gate for a new dataset via the library API."""
        self.projections[name] = nn.Sequential(
            nn.Linear(input_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.GELU()
        )
    
    @staticmethod
    def normalize(x):
        """
        Standardizes input features (Z-Score Normalization).
        Critical for the Tunneling Stabilizer (tanh) to function correctly.
        Formula: (x - mean) / (std + 1e-6)
        """
        if isinstance(x, torch.Tensor):
            return (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
        else:
            # Assume numpy
            import numpy as np
            return (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-6)

    def forward(self, x, source=None, era=None):
        # Support legacy 'era' argument
        target_source = source if source else era
        
        # Route to appropriate gate
        if not target_source or target_source not in self.projections:
            raise ValueError(f"Unknown source: {target_source}. Please input valid source.")
        
        # 1. Project
        x = self.projections[target_source](x)
        
        # 2. Manifold & Tunneling (Residual)
        latent_res = self.manifold(x)
        latent = x + latent_res
        stabilized = self.tunnel(latent)
        
        # 3. Decision
        logits = self.head(stabilized)
        
        if self.training:
            return logits

        # Inference: Clarity Snap Logic
        probs = torch.softmax(logits, dim=1)
        conf_mask = probs.max(dim=1).values > 0.85
        if conf_mask.any():
            expanded_mask = conf_mask.unsqueeze(1).expand_as(probs)
            indices = probs.argmax(dim=1)
            one_hot = torch.zeros_like(probs)
            one_hot.scatter_(1, indices.unsqueeze(1), 1.0)
            probs = torch.where(expanded_mask, one_hot, probs)
            
        return probs
