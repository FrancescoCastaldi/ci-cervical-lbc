import torch
import torch.nn.functional as F


def tv_loss(x):
    diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
    diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]
    return diff_h.abs().mean() + diff_w.abs().mean()


def gaussian_kernel_tensor(kernel_size=9, sigma=2.0, channels=3):
    ax = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    k = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    k = k / k.sum()
    k = k.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    return k


def apply_blur(x, kernel, kernel_size):
    pad = kernel_size // 2
    return F.conv2d(x, kernel, padding=pad, groups=x.shape[1])


def tv_restore(degraded, kernel_size=9, sigma=2.0,
               lambda_reg=0.005, max_iter=150, lr=0.001):
    """
    Ricostruzione TV con deblur.
    
    Minimizza: ||H*x - y||² + lambda_reg * TV(x)
    
    Input/output in range [-1, 1].
    """
    channels = degraded.shape[0]
    kernel = gaussian_kernel_tensor(kernel_size, sigma, channels)

    x = degraded.clone().unsqueeze(0).requires_grad_(True)  # [1, C, H, W]
    y = degraded.unsqueeze(0)                                # [1, C, H, W]

    optimizer = torch.optim.Adam([x], lr=lr)

    for _ in range(max_iter):
        optimizer.zero_grad()
        blurred_x = apply_blur(x, kernel, kernel_size)
        data_fidelity = F.mse_loss(blurred_x, y)
        reg = tv_loss(x)
        loss = data_fidelity + lambda_reg * reg
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x.clamp_(-1.0, 1.0)  # range [-1, 1]

    return x.detach().squeeze(0)