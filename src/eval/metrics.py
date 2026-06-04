import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def to_numpy(tensor):
    return tensor.detach().cpu().permute(1, 2, 0).numpy()


def compute_psnr(pred, gt):
    pred_np = to_numpy(pred.clamp(0, 1))
    gt_np = to_numpy(gt.clamp(0, 1))
    return psnr(gt_np, pred_np, data_range=1.0)


def compute_ssim(pred, gt):
    pred_np = to_numpy(pred.clamp(0, 1))
    gt_np = to_numpy(gt.clamp(0, 1))
    return ssim(gt_np, pred_np, data_range=1.0, channel_axis=-1)


def evaluate(pred, gt):
    return {
        "psnr": compute_psnr(pred, gt),
        "ssim": compute_ssim(pred, gt),
    }
