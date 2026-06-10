import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from .model import LightUNet


_model_cache = {}


def _load_model(weights_path, device):
    key = (str(weights_path), str(device))
    if key in _model_cache:
        return _model_cache[key]

    checkpoint = torch.load(str(weights_path), map_location="cpu")

    model = LightUNet(in_channels=3, out_channels=3, base_channels=32)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(device)

    schedule = {
        "num_timesteps": checkpoint["num_timesteps"],
        "betas": checkpoint["betas"],
        "alphas": checkpoint["alphas"],
        "alphas_cumprod": checkpoint["alphas_cumprod"],
    }

    _model_cache[key] = (model, schedule)
    return model, schedule


def _psf_to_otf(kernel, img_size):
    pad_h = img_size[-2] - kernel.shape[-2]
    pad_w = img_size[-1] - kernel.shape[-1]
    pad_h_before = pad_h // 2
    pad_h_after = pad_h - pad_h_before
    pad_w_before = pad_w // 2
    pad_w_after = pad_w - pad_w_before
    padded = F.pad(kernel, (pad_w_before, pad_w_after, pad_h_before, pad_h_after))
    shifted = torch.roll(padded, (-kernel.shape[-2] // 2, -kernel.shape[-1] // 2), dims=(-2, -1))
    otf = torch.fft.fftn(shifted, dim=(-2, -1))
    return otf


def _data_fidelity_fft(y, x0_pred, kernel_otf, rho):
    FB = kernel_otf
    FBC = torch.conj(FB)
    F2B = torch.abs(FB) ** 2

    y_f = torch.fft.fftn(y, dim=(-2, -1))
    x0_f = torch.fft.fftn(x0_pred, dim=(-2, -1))

    numerator = FBC * y_f + rho * x0_f
    denominator = F2B + rho

    x_f = numerator / denominator
    x = torch.fft.ifftn(x_f, dim=(-2, -1)).real
    return x


def run_diffpir(
    degraded,
    num_steps=15,
    noise_level=0.05,
    weights_path=None,
    kernel_size=9,
    blur_sigma=2.0,
    lambda_=1.0,
    zeta=0.1,
    t_start=None,
    return_timing=False,
):
    """
    DiffPIR: Diffusion-based Plug-and-Play Image Restoration.

    Uses a lightweight DDPM model trained on cervical cancer images,
    combined with FFT-based data fidelity for deblurring.

    Args:
        degraded: torch.Tensor [C, H, W] in [-1, 1]
        num_steps: number of diffusion sampling steps (subsampled from 0..t_start)
        noise_level: AWGN noise level
        weights_path: path to trained DDPM .pt file
        kernel_size: Gaussian blur kernel size
        blur_sigma: Gaussian blur sigma
        lambda_: data-fidelity weight
        zeta: stochasticity parameter (0=deterministic, 1=fully stochastic)
        t_start: starting timestep (None = auto from noise schedule)
        return_timing: if True, returns (restored, inference_time)

    Returns:
        restored: torch.Tensor [C, H, W] in [0, 1]
    """
    start_time = time.time()

    device = degraded.device if degraded.is_cuda else torch.device("cpu")
    y = degraded.unsqueeze(0).to(device)

    if weights_path is None:
        weights_path = Path(__file__).resolve().parent / "weights" / "ddpm_lbc.pt"
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"DDPM weights not found at {weights_path}\n"
            f"Run: python -m src.methods.diffpir.train"
        )

    model, schedule = _load_model(weights_path, device)
    num_timesteps = schedule["num_timesteps"]
    betas = schedule["betas"].to(device)
    alphas_cumprod = schedule["alphas_cumprod"].to(device)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # --- Gaussian blur kernel OTF ---
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * blur_sigma ** 2))
    g = g / g.sum()
    k = g[:, None] * g[None, :]
    k = k / k.sum()
    k_tensor = k.unsqueeze(0).unsqueeze(0).to(device)
    img_size = y.shape[-2:]
    kernel_otf = _psf_to_otf(k_tensor, img_size)

    # --- Find t_y (timestep matching measurement noise level) ---
    sigma_ks = sqrt_1m_alphas_cumprod / sqrt_alphas_cumprod
    t_y = torch.argmin(torch.abs(sigma_ks - 2 * noise_level)).item()

    # --- Choose t_start where x0 estimation is numerically stable ---
    #  x0 = (x_t - sqrt(1-α̅) * ε_θ) / sqrt(α̅)
    #  Error amplification = sqrt(1-α̅) / sqrt(α̅)
    #  We want α̅ large enough that this amplification is manageable (< 5).
    STABLE_ALPHA_THRESH = 0.08
    if t_start is None:
        stable_mask = alphas_cumprod >= STABLE_ALPHA_THRESH
        if stable_mask.any():
            t_start = torch.where(stable_mask)[0][-1].item()
        else:
            t_start = min(200, num_timesteps - 1)
    else:
        t_start = min(t_start, num_timesteps - 1)

    if t_y >= t_start:
        t_start = min(t_y + 5, num_timesteps - 1)

    # --- Initialize x at t_start with correct noise level ---
    ac_start = sqrt_alphas_cumprod[t_start]
    ac_t_y = sqrt_alphas_cumprod[t_y]
    alpha_eff = ac_start / ac_t_y
    var_eff = sqrt_1m_alphas_cumprod[t_start] ** 2 - alpha_eff ** 2 * sqrt_1m_alphas_cumprod[t_y] ** 2
    var_eff = var_eff.clamp(min=0.0)
    x = alpha_eff * y + torch.sqrt(var_eff) * torch.randn_like(y)

    # --- ρ_t schedule: data-fidelity weight ---
    #  ρ_t = λ * σ² / σ_k(t)²
    #  where σ_k(t) = sqrt(1-α̅(t)) / sqrt(α̅(t))
    sigma = max(0.001, noise_level)
    rhos = lambda_ * (sigma ** 2) / (sigma_ks ** 2)

    # --- Build subsampled sequence: t_start → 0 with sqrt spacing ---
    if num_steps <= 1:
        seq = [0]
    else:
        seq = np.sqrt(np.linspace(0, t_start ** 2, num_steps))
        seq = sorted(set(int(s) for s in seq))
        if seq[-1] != t_start:
            seq.append(t_start)
        seq = seq[::-1]

    # --- Sampling loop ---
    for i in range(len(seq)):
        t_i = seq[i]

        with torch.no_grad():
            t_tensor = torch.tensor([t_i], device=device).expand(y.shape[0])
            pred_noise = model(x, t_tensor)

        # Estimate x0 from current x and predicted noise
        x0 = (x - sqrt_1m_alphas_cumprod[t_i] * pred_noise) / sqrt_alphas_cumprod[t_i]
        x0 = x0.clamp(-1, 1)

        # Data fidelity (skip on last step)
        if i < len(seq) - 1:
            rho_t = rhos[t_i].reshape(1, 1, 1, 1)
            x0 = _data_fidelity_fft(y, x0, kernel_otf, rho_t)

        # DDIM step to next timestep (or finalize)
        if i < len(seq) - 1:
            t_im1 = seq[i + 1]
            eps = (x - sqrt_alphas_cumprod[t_i] * x0) / sqrt_1m_alphas_cumprod[t_i]

            # Deterministic DDIM component
            x_ddpm_mean = sqrt_alphas_cumprod[t_im1] * x0
            x_ddpm_dir = torch.sqrt(sqrt_1m_alphas_cumprod[t_im1] ** 2) * eps

            # Stochastic mixing controlled by zeta
            noise_std = sqrt_1m_alphas_cumprod[t_im1]
            x = x_ddpm_mean + np.sqrt(1 - zeta) * x_ddpm_dir + np.sqrt(zeta) * noise_std * torch.randn_like(x)
        else:
            x = x0

    # Output in [-1, 1] to match evaluate()/to_numpy() expectation
    result = x.squeeze(0).clamp(-1, 1).cpu()

    if return_timing:
        return result, time.time() - start_time
    return result
