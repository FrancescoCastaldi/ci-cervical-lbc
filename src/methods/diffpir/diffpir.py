# DiffPir integration wrapper
# Reference: https://github.com/yuanzhi-zhu/DiffPIR
# Requires: clone DiffPIR repo and place pretrained weights in src/methods/diffpir/weights/

import torch
from pathlib import Path


def run_diffpir(degraded, num_steps=100, noise_level=0.05, weights_path=None):
    """
    Wrapper for DiffPir inference.
    degraded: torch.Tensor [C, H, W] in [0,1]
    Returns: restored tensor [C, H, W]
    """
    raise NotImplementedError(
        "Integrate DiffPIR from https://github.com/yuanzhi-zhu/DiffPIR.\n"
        "Clone the repo, download pretrained weights, and adapt this wrapper."
    )
