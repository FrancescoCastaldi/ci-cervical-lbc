import sys
import time
import torch
import numpy as np
from pathlib import Path

DIFFPIR_ROOT = Path(__file__).resolve().parents[3] / "external" / "diffpir"

_model_cache = {}


def _setup_path():
    if not DIFFPIR_ROOT.exists():
        raise FileNotFoundError(
            f"DiffPIR non trovato in {DIFFPIR_ROOT}\n"
            f"Esegui: git clone https://github.com/yuanzhi-zhu/DiffPIR.git external/diffpir"
        )
    root = str(DIFFPIR_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_model(weights_path, device):
    key = (str(weights_path), str(device))
    if key in _model_cache:
        return _model_cache[key]

    _setup_path()
    from guided_diffusion.script_util import (
        model_and_diffusion_defaults,
        create_model_and_diffusion,
        args_to_dict,
    )
    from utils import utils_model

    model_config = dict(
        model_path=str(weights_path),
        num_channels=256,
        num_res_blocks=2,
        attention_resolutions="8,16,32",
    )
    args = utils_model.create_argparser(model_config).parse_args([])
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(device)

    _model_cache[key] = (model, diffusion)
    return model, diffusion


def run_diffpir(
    degraded,
    num_steps=100,
    noise_level=0.05,
    weights_path=None,
    kernel_size=9,
    blur_sigma=2.0,
    lambda_=1.0,
    zeta=0.1,
    return_timing=False,
):
    """
    DiffPIR deblur+denoise wrapper.

    Args:
        degraded: torch.Tensor [C, H, W] in [-1, 1]
        num_steps: number of diffusion sampling steps
        noise_level: AWGN noise level
        weights_path: path to pretrained .pt file
        kernel_size: Gaussian blur kernel size
        blur_sigma: Gaussian blur sigma
        lambda_: data-fidelity weight
        zeta: stochasticity parameter
        return_timing: if True, returns (restored, inference_time)

    Returns:
        restored: torch.Tensor [C, H, W] in [0, 1]
        or (restored, inference_time) if return_timing=True
    """
    start_time = time.time()
    _setup_path()
    from utils import utils_model
    from utils import utils_sisr as sr

    device = degraded.device if degraded.is_cuda else torch.device("cpu")

    if weights_path is None:
        weights_path = (
            Path(__file__).resolve().parent
            / "weights"
            / "256x256_diffusion_uncond.pt"
        )
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Pesi non trovati: {weights_path}\n"
            f"Scarica da: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt\n"
            f"E salva in: {weights_path}"
        )

    y = (degraded.clamp(-1, 1) + 1) / 2
    y = y.unsqueeze(0).to(device)

    model, diffusion = _load_model(weights_path, device)

    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords**2) / (2 * blur_sigma**2))
    g = g / g.sum()
    k = g[:, None] * g[None, :]
    k = k / k.sum()
    k_np = k.numpy()
    k_tensor = k.unsqueeze(0).unsqueeze(0).to(device)

    num_train_timesteps = 1000
    beta_start = 0.1 / 1000
    beta_end = 20 / 1000
    betas = torch.linspace(
        beta_start, beta_end, num_train_timesteps, dtype=torch.float32, device=device
    )
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    reduced_alpha_cumprod = sqrt_1m_alphas_cumprod / sqrt_alphas_cumprod

    reduced_np = reduced_alpha_cumprod.cpu().numpy()

    sigma = max(0.001, noise_level)
    sf = 1

    FB, FBC, F2B, FBFy = sr.pre_calculate(y, k_tensor, sf)

    sigma_ks = sqrt_1m_alphas_cumprod / sqrt_alphas_cumprod
    rhos = lambda_ * (sigma**2) / (sigma_ks**2)

    t_start = num_train_timesteps - 1
    t_y = utils_model.find_nearest(reduced_np, 2 * noise_level)
    sqrt_alpha_effective = sqrt_alphas_cumprod[t_start] / sqrt_alphas_cumprod[t_y]
    x = sqrt_alpha_effective * (2 * y - 1) + torch.sqrt(
        sqrt_1m_alphas_cumprod[t_start] ** 2
        - sqrt_alpha_effective**2 * sqrt_1m_alphas_cumprod[t_y] ** 2
    ) * torch.randn_like(y)

    seq = np.sqrt(np.linspace(0, num_train_timesteps**2, num_steps))
    seq = [int(s) for s in seq]
    seq[-1] = seq[-1] - 1

    sigmas_arr = reduced_np[::-1].copy()

    for i in range(len(seq)):
        curr_sigma = float(sigmas_arr[seq[i]])
        t_i = utils_model.find_nearest(reduced_np, curr_sigma)
        if t_i > t_start:
            continue

        x0 = utils_model.model_fn(
            x,
            noise_level=curr_sigma * 255,
            model_out_type="pred_xstart",
            model_diffusion=model,
            diffusion=diffusion,
            ddim_sample=False,
            alphas_cumprod=alphas_cumprod,
        )

        if seq[i] != seq[-1]:
            tau = rhos[t_i].reshape(1, 1, 1, 1)
            x0_p = x0 / 2 + 0.5
            x0_p = sr.data_solution(x0_p.float(), FB, FBC, F2B, FBFy, tau, sf)
            x0_p = x0_p * 2 - 1
            x0 = x0 + 1.0 * (x0_p - x0)

        if i < len(seq) - 1:
            next_sigma = float(sigmas_arr[seq[i + 1]])
            t_im1 = utils_model.find_nearest(reduced_np, next_sigma)
            eps = (x - sqrt_alphas_cumprod[t_i] * x0) / sqrt_1m_alphas_cumprod[t_i]
            eta = 0.0
            eta_sigma = (
                eta
                * sqrt_1m_alphas_cumprod[t_im1]
                / sqrt_1m_alphas_cumprod[t_i]
                * torch.sqrt(betas[t_i])
            )
            x = sqrt_alphas_cumprod[t_im1] * x0 + np.sqrt(1 - zeta) * (
                torch.sqrt(
                    sqrt_1m_alphas_cumprod[t_im1] ** 2 - eta_sigma**2
                )
                * eps
                + eta_sigma * torch.randn_like(x)
            ) + np.sqrt(zeta) * sqrt_1m_alphas_cumprod[t_im1] * torch.randn_like(x)

    result = (x / 2 + 0.5).clamp(0, 1).squeeze(0).cpu()
    
    if return_timing:
        inference_time = time.time() - start_time
        return result, inference_time
    return result
