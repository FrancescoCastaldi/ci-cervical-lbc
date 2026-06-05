import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def to_numpy(tensor):
    """Convert tensor [-1,1] to numpy [0,1]."""
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = img * 0.5 + 0.5
    return np.clip(img, 0.0, 1.0)


def compute_psnr(pred, gt):
    return psnr(to_numpy(gt), to_numpy(pred), data_range=1.0)


def compute_ssim(pred, gt):
    return ssim(to_numpy(gt), to_numpy(pred), data_range=1.0, channel_axis=-1)


def evaluate(pred, gt):
    return {
        "psnr": compute_psnr(pred, gt),
        "ssim": compute_ssim(pred, gt),
    }