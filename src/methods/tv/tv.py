import torch
import torch.nn.functional as F


def tv_loss(img):
    diff_h = img[:, :, 1:, :] - img[:, :, :-1, :]
    diff_w = img[:, :, :, 1:] - img[:, :, :, :-1]
    return diff_h.abs().mean() + diff_w.abs().mean()


def tv_restore(degraded, lambda_reg=0.1, max_iter=300, lr=0.01):
    x = degraded.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)

    for _ in range(max_iter):
        optimizer.zero_grad()
        data_fidelity = F.mse_loss(x, degraded)
        reg = tv_loss(x.unsqueeze(0))
        loss = data_fidelity + lambda_reg * reg
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x.clamp_(0.0, 1.0)

    return x.detach()
