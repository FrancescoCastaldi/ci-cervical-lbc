"""Shared utilities for device detection and other common operations."""

import torch


def get_device(verbose: bool = True) -> torch.device:
    """Detect the best available device: CUDA > MPS > CPU.

    Args:
        verbose: If True, prints which device was selected.

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"[Device] CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print("[Device] Apple MPS")
    else:
        device = torch.device("cpu")
        if verbose:
            print("[Device] CPU")
    return device
