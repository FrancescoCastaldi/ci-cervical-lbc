"""Unit tests for TV method (variational deblur + denoise)."""
import torch
from src.methods.tv.tv import tv_loss, tv_restore, gaussian_kernel_tensor, apply_blur


def test_tv_loss_nonnegative():
    """TV loss is always non-negative."""
    x = torch.randn(1, 3, 32, 32)
    loss = tv_loss(x)
    assert loss >= 0, f"TV loss should be >= 0, got {loss}"


def test_tv_loss_uniform_image():
    """TV loss is zero for a perfectly uniform image."""
    x = torch.ones(1, 3, 32, 32)
    loss = tv_loss(x)
    assert loss == 0, f"Uniform image should have TV=0, got {loss}"


def test_tv_loss_high_variance():
    """Noisy image has higher TV loss than smooth image."""
    smooth = torch.zeros(1, 3, 32, 32)
    noisy = torch.randn(1, 3, 32, 32) * 0.5
    loss_smooth = tv_loss(smooth)
    loss_noisy = tv_loss(noisy)
    assert loss_noisy > loss_smooth, \
        f"Noisy (loss={loss_noisy}) should be > smooth (loss={loss_smooth})"


def test_gaussian_kernel_tensor_shape():
    """Kernel has correct shape and sums to 1."""
    kernel = gaussian_kernel_tensor(kernel_size=9, sigma=2.0, channels=3)
    assert kernel.shape == (3, 1, 9, 9), f"Expected (3,1,9,9), got {kernel.shape}"
    for c in range(3):
        assert abs(kernel[c].sum().item() - 1.0) < 1e-5, \
            f"Kernel channel {c} does not sum to 1"


def test_apply_blur_preserves_shape():
    """Blur preserves spatial dimensions with same padding."""
    x = torch.randn(1, 3, 32, 32)
    kernel = gaussian_kernel_tensor(9, 2.0, 3)
    out = apply_blur(x, kernel, 9)
    assert out.shape == (1, 3, 32, 32), f"Expected (1,3,32,32), got {out.shape}"


def test_tv_restore_output_range():
    """Restored image is in [-1, 1] range."""
    degraded = torch.randn(3, 32, 32) * 0.1
    restored = tv_restore(degraded, kernel_size=5, sigma=1.0,
                          lambda_reg=0.01, max_iter=20)
    assert torch.isfinite(restored).all(), "Output contains NaN or Inf"
    assert restored.min() >= -1.0 and restored.max() <= 1.0, \
        f"Output range [{restored.min():.3f}, {restored.max():.3f}] outside [-1,1]"


def test_tv_restore_output_shape():
    """Restored image has same shape as input."""
    degraded = torch.randn(3, 64, 64) * 0.05
    restored = tv_restore(degraded, kernel_size=5, sigma=1.0,
                          lambda_reg=0.01, max_iter=10)
    assert restored.shape == degraded.shape, \
        f"Expected {degraded.shape}, got {restored.shape}"


def test_tv_restore_gradient_flow():
    """Gradient flows through the optimization (loss decreases)."""
    degraded = torch.randn(3, 32, 32) * 0.1
    x = degraded.clone().unsqueeze(0).requires_grad_(True)
    y = degraded.unsqueeze(0)
    kernel = gaussian_kernel_tensor(5, 1.0, 3)
    optimizer = torch.optim.Adam([x], lr=0.001)

    loss_before = None
    for i in range(30):
        optimizer.zero_grad()
        blurred_x = apply_blur(x, kernel, 5)
        data_fidelity = torch.nn.functional.mse_loss(blurred_x, y)
        reg = tv_loss(x)
        loss = data_fidelity + 0.01 * reg
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x.clamp_(-1.0, 1.0)
        if i == 0:
            loss_before = loss.item()

    loss_after = loss.item()
    assert loss_after < loss_before, \
        f"Loss did not decrease: {loss_before:.4f} -> {loss_after:.4f}"
