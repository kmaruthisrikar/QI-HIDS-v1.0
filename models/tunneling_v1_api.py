"""
QI-HIDS v1.0: TUNNELING LEARNING INFERENCE WRAPPER
Encapsulates the trained Tunneling Learning Manifold (.pth) into a high-level API for deployment.
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import torch
import torch.nn as nn
from models.v1_engine import build_v1

class TunnelingLearningInference:
    def __init__(self, pth_path="models/tunneling_v1.pth", k_dim=36, c_dim=78):
        self.device = torch.device("cpu")
        self.model = build_v1(legacy_dim=k_dim, modern_dim=c_dim).to(self.device)
        
        # Load Weights
        try:
            state_dict = torch.load(pth_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"✓ Tunneling Learning Engine v1 initialized from: {pth_path}")
            print(f"  Architectural State: [KDD={k_dim}, CIC={c_dim}]")
        except Exception as e:
            raise RuntimeError(f"FAILED to initialize Tunneling Learning manifold from weights: {e}")

    def detect(self, feature_vector, era='modern'):
        """
        Run inference on a single flow or batch.
        era: 'legacy' (36 dim) or 'modern' (78 dim)
        """
        # Auto-convert to tensor
        if not torch.is_tensor(feature_vector):
            feature_vector = torch.tensor(feature_vector, dtype=torch.float32)
            
        # Ensure correct shape [Batch, Features]
        if len(feature_vector.shape) == 1:
            feature_vector = feature_vector.unsqueeze(0)
            
        with torch.no_grad():
            outputs = self.model(feature_vector, era=era)
            # v1_engine already returns probabilities (Clarity Snapped)
            
            # Extract Results
            confidences, classes = torch.max(outputs, dim=1)
            
            results = []
            for conf, cls in zip(confidences, classes):
                label = "MALICIOUS" if cls.item() == 1 else "NORMAL"
                results.append({
                    "prediction": label,
                    "confidence": f"{conf.item():.2%}",
                    "status": "CRITICAL" if label == "MALICIOUS" else "SAFE"
                })
            
            return results

    def get_manifold_state(self, x, era='modern'):
        """
        Returns the latent space projection (128-dim) before decision.
        Useful for behavioral visualization.
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        if len(x.shape) == 1: x = x.unsqueeze(0)
            
        with torch.no_grad():
            if era == 'legacy':
                x = self.model.legacy_proj(x)
            else:
                x = self.model.modern_proj(x)
                
            latent_res = self.model.manifold(x)
            latent = x + latent_res
            stabilized = self.model.tunnel(latent)
            
            return stabilized

if __name__ == "__main__":
    # Example usage
    try:
        engine = TunnelingLearningInference()
        dummy_flow = [0.0] * 78 # Modern dummy
        prediction = engine.detect(dummy_flow)
        print("\nFlow Diagnostic Results:")
        print(prediction)
    except Exception as e:
        print(f"Error: {e}")
