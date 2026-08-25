"""
dia/hardware.py
───────────────
Hardware detection and training estimation utilities.
"""

import os
import subprocess
import time

def get_cpu_cores() -> int:
    """Returns the total number of logical CPU cores."""
    return os.cpu_count() or 4

def is_gpu_available() -> bool:
    """Detects if an NVIDIA GPU is available via nvidia-smi."""
    try:
        # Check if nvidia-smi runs successfully
        result = subprocess.run(
            ["nvidia-smi"], 
            capture_output=True, 
            text=True, 
            timeout=2
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Fallback to PyTorch/TensorFlow if installed, though we want this to be lightweight
    try:
        import torch # type: ignore
        return torch.cuda.is_available()
    except ImportError:
        pass
        
    return False

def estimate_training_time(n_rows: int, n_cols: int, n_models: int, use_gpu: bool) -> str:
    """
    Provides a rough human-readable string estimate of training time
    based on data size and selected model count.
    """
    # Extremely rough heuristic:
    # A 100k x 20 dataset takes ~5s per tree model on CPU.
    # GPU reduces time by ~3x for tree models.
    
    complexity_factor = (n_rows * n_cols) / 100_000
    base_time_seconds = complexity_factor * n_models * 0.5
    
    if use_gpu:
        base_time_seconds *= 0.4
        
    # Minimum overhead
    base_time_seconds += 2 
    
    if base_time_seconds < 10:
        return "< 10 seconds"
    elif base_time_seconds < 30:
        return "~ 30 seconds"
    elif base_time_seconds < 60:
        return "~ 1 minute"
    elif base_time_seconds < 300:
        minutes = int(base_time_seconds // 60)
        return f"~ {minutes} minutes"
    else:
        return "> 5 minutes (Grab a coffee ☕)"
