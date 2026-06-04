import torch
import torch.nn.functional as F


def gaussian_kernel(kernel_size=9, sigma=2.0):
    coords = torch.arange(kernel_size).float() - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]
    return kernel / kernel.sum()


def apply_blur(img, kernel_size=9, sigma=2.0):
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(img.shape[0], 1, 1, 1)
    padding = kernel_size // 2
    blurred = F.conv2d(img.unsqueeze(0), kernel, padding=padding, groups=img.shape[0])
    return blurred.squeeze(0)


def apply_noise(img, noise_level=0.01):
    noise = torch.randn_like(img) * noise_level
    return torch.clamp(img + noise, -1.0, 1.0)


def degrade(img, kernel_size=9, sigma=2.0, noise_level=0.01):
    blurred = apply_blur(img, kernel_size, sigma)
    degraded = apply_noise(blurred, noise_level)
    return degraded
