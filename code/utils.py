##########################################
# Shared Utility Functions
##########################################

import torch

def setup_device():
    """Setup computation device (MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def get_version_label(version):
    """Convert version suffix to readable label."""
    version_map = {
        1: 'Standard System Prompt',
        2: 'Force True',
        3: 'Force False'
    }
    return version_map.get(version, version)
