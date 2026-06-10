"""Unit tests for UNet architecture (snella, ~500K params)."""
import torch
import pytest
from src.methods.unet.unet import UNet, DoubleConv


def test_unet_forward_shape():
    """UNet forward pass preserves input spatial dimensions."""
    model = UNet(in_channels=4, out_channels=3)
    x = torch.randn(1, 4, 256, 256)
    out = model(x)
    assert out.shape == (1, 3, 256, 256), f"Expected (1,3,256,256), got {out.shape}"


def test_unet_gradient_flow():
    """Gradients flow through all parameters and are not all zero."""
    model = UNet(in_channels=4, out_channels=3)
    x = torch.randn(2, 4, 128, 128)
    target = torch.randn(2, 3, 128, 128)
    out = model(x)
    loss = ((out - target) ** 2).mean()
    loss.backward()

    grads = []
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No grad for {name}"
        grads.append(param.grad.abs().mean().item())

    assert len(grads) > 0
    assert sum(grads) > 0, "All gradients are zero — dead model"


def test_unet_output_finite():
    """UNet output is finite and in [-1, 1] range (tanh output)."""
    model = UNet()
    x = torch.randn(1, 4, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out).all(), "Output contains NaN or Inf"
    assert out.min() >= -1.0 and out.max() <= 1.0, \
        f"Output range [{out.min().item():.3f}, {out.max().item():.3f}] outside [-1,1]"


def test_unet_params():
    """UNet has ~2M parameters with noise conditioning (was 31M)."""
    model = UNet()
    n_params = sum(p.numel() for p in model.parameters())
    # ~1.94M with 4 input channels (RGB+noise) + GroupNorm + bottleneck
    assert 1_500_000 <= n_params <= 2_500_000, \
        f"Expected ~2M params, got {n_params:,}. Original was 31M."


def test_doubleconv():
    """DoubleConv submodule works with GroupNorm."""
    conv = DoubleConv(4, 16, num_groups=4)
    x = torch.randn(2, 4, 32, 32)
    out = conv(x)
    assert out.shape == (2, 16, 32, 32), f"Expected (2,16,32,32), got {out.shape}"
    assert torch.isfinite(out).all()


def test_multi_seed_diversity():
    """Different random initializations produce different outputs."""
    x = torch.randn(1, 4, 128, 128)
    outputs = []
    for seed in range(5):
        torch.manual_seed(seed)
        model = UNet()
        with torch.no_grad():
            out = model(x)
        outputs.append(out)
    stacked = torch.stack(outputs)
    std_across = stacked.std(dim=0).mean().item()
    assert std_across > 0.001, f"Outputs too similar (std={std_across})"


def test_batch_independence():
    """Each image in a batch is processed independently."""
    model = UNet()
    img1 = torch.randn(1, 4, 128, 128)

    # Same image twice in batch
    img_batch = torch.cat([img1, img1], dim=0)
    with torch.no_grad():
        out_batch = model(img_batch)
    assert torch.allclose(out_batch[0], out_batch[1], atol=1e-6), \
        "Identical inputs in batch produced different outputs"


def test_noise_conditioning_effect():
    """Different noise level maps produce different outputs."""
    model = UNet()
    base = torch.randn(1, 3, 64, 64)
    noise_map_low = torch.full((1, 1, 64, 64), 0.005)
    noise_map_high = torch.full((1, 1, 64, 64), 0.1)

    x_low = torch.cat([base, noise_map_low], dim=1)
    x_high = torch.cat([base, noise_map_high], dim=1)

    with torch.no_grad():
        out_low = model(x_low)
        out_high = model(x_high)

    diff = (out_low - out_high).abs().mean().item()
    assert diff > 0.001, \
        f"Noise conditioning not affecting output (diff={diff})"
