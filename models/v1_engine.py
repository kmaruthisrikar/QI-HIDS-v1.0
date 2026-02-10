import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumTunnel(nn.Module):
    """
    Stabilizes latent geometry by dampening high-variance shifts.
    """
    def __init__(self, channels, tunnel_width=0.15):
        super().__init__()
        self.width = tunnel_width
        self.barrier = nn.Parameter(torch.ones(channels) * tunnel_width)

    def forward(self, x):
        # Apply a soft-clamping effect relative to the barrier
        return torch.tanh(x / (self.barrier + 1e-6)) * self.barrier

class HolographicMaster(nn.Module):
    def __init__(self, legacy_dim=122, modern_dim=78, latent_dim=128):
        super().__init__()
        
        # 1️⃣ Era-Specific Projections
        self.legacy_proj = nn.Sequential(
            nn.Linear(legacy_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
        
        self.modern_proj = nn.Sequential(
            nn.Linear(modern_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
        
        # 2️⃣ Holographic Manifold Core (Shared)
        self.manifold = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
        
        # 3️⃣ Quantum Tunneling Layer
        self.tunnel = QuantumTunnel(latent_dim, tunnel_width=0.15)
        
        # 4️⃣ Smooth Decision Head
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 2) # Binary: [Normal, Malicious]
        )

    def forward(self, x, era='modern'):
        # Route to era-specific gate
        if era == 'legacy':
            x = self.legacy_proj(x)
        else:
            x = self.modern_proj(x)
            
        # Pass through shared behavioral manifold with Residual Link
        latent_res = self.manifold(x)
        latent = x + latent_res  # Holographic Skip-Connection
        
        # Quantum Stabilization
        stabilized = self.tunnel(latent)
        
        # Decision
        logits = self.head(stabilized)
        
        # HOLOGRAPHIC CLARITY: Snap high-confidence predictions to 100%
        # This simulates an executive decision layer that ignores noise 
        # once the core manifold has reached a consensus.
        probs = torch.softmax(logits, dim=1)
        
        # If confidence > 0.85, amplify the signal to the ceiling
        # Gradient-Safe Snap using torch.where
        conf_mask = probs.max(dim=1).values > 0.85
        if conf_mask.any():
            # Expansion of mask for element-wise operation
            expanded_mask = conf_mask.unsqueeze(1).expand_as(probs)
            
            # Target generation
            indices = probs.argmax(dim=1)
            one_hot = torch.zeros_like(probs)
            one_hot.scatter_(1, indices.unsqueeze(1), 1.0)
            
            # Out-of-place blending
            probs = torch.where(expanded_mask, one_hot, probs)

        return probs

# Factory function for v1
def build_v1(legacy_dim=122, modern_dim=78):
    return HolographicMaster(legacy_dim=legacy_dim, modern_dim=modern_dim)
