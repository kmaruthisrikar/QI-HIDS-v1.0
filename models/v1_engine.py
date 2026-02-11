
import torch
import torch.nn as nn
from models.tunneling_lib import UniversalTunnelingNetwork

# This file now acts as a Production Adapter for the Universal Library.
# It ensures backward compatibility while using the powerful extensible core.

def build_v1(legacy_dim=122, modern_dim=78):
    """
    Factory function that initializes the Universal Tunneling Network
    with the standard V1 configuration (Legacy + Modern gates).
    """
    # Initialize the Universal Core
    model = UniversalTunnelingNetwork(latent_dim=128)
    
    # Register the Standard Gates
    # This proves the library is working: we 'load' the datasets into the engine here.
    model.add_dataset('legacy', input_dim=legacy_dim)
    model.add_dataset('modern', input_dim=modern_dim)
    
    return model

# Alias for type-checking if needed elsewhere
TunnelingLearningEngine = UniversalTunnelingNetwork
