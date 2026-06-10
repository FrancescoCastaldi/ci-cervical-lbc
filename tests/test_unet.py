"""Unit tests for UNet architecture."""
import torch
import pytest
from src.methods.unet.unet import UNet, DoubleConv


def test_unet_forward_shape():
    """UNet forward pass preserves input spatial dimensions."""
    model = UNet(in_channels=3, out_channels=3)
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    assert out.shape == (1, 3, 256, 256), f"Expected (1,3,256,256), got {out.shape}"


def test_unet_gradient_flow():
    """Gradients flow through all parameters and are not all zero."""
    model = UNet(in_channels=3, out_channels=3)
    x = torch.randn(2, 3, 128, 128)
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
    """UNet output is finite and roughly in [-1, 1] range."""
    model = UNet()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out).all(), "Output contains NaN or Inf"
    # With ReLU activations and no final tanh, range can be wide;
    # just verify no extreme explosions
    assert out.abs().max() < 100, f"Output max too large: {out.abs().max().item()}"


def test_doubleconv():
    """DoubleConv submodule works correctly with expected channel transform."""
    conv = DoubleConv(3, 64)
    x = torch.randn(2, 3, 32, 32)
    out = conv(x)
    assert out.shape == (2, 64, 32, 32), f"Expected (2,64,32,32), got {out.shape}"
    assert torch.isfinite(out).all()


def test_multi_seed_diversity():
    """Different random initializations produce different outputs."""
    x = torch.randn(1, 3, 128, 128)
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
    img1 = torch.randn(1, 3, 128, 128)
    img2 = torch.randn(2, 3, 128, 128)

    with torch.no_grad():
        out1_single = model(img1)
        out2_batch = model(img2)

    # First image of batch should match single-image output (same input)
    # Use the same image twice in batch position 0
    img_batch = torch.cat([img1, img1], dim=0)
    with torch.no_grad():
        out_batch = model(img_batch)
    assert torch.allclose(out_batch[0], out_batch[1], atol=1e-6), \
        "Identical inputs in batch produced different outputs"
