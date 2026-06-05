"""Tests for DiffPIR algorithm (data fidelity, OTF, model loading)."""

import torch
from pathlib import Path
from src.methods.diffpir.diffpir import (
    _psf_to_otf,
    _data_fidelity_fft,
    _load_model,
)


class TestPSFtoOTF:
    def test_output_shape(self):
        """OTF should match the image size."""
        kernel = torch.ones(1, 1, 9, 9) / 81.0
        img_size = (64, 64)
        otf = _psf_to_otf(kernel, img_size)
        assert otf.shape == (1, 1, 64, 64), f"Got {otf.shape}"

    def test_dc_component(self):
        """DC component of OTF should be 1 for normalized kernel."""
        kernel = torch.ones(1, 1, 9, 9) / 81.0
        img_size = (64, 64)
        otf = _psf_to_otf(kernel, img_size)
        # The sum of the PSF (all 1/81 * 81 = 1) should appear as DC = 1
        assert abs(otf[..., 0, 0].imag.item()) < 1e-6
        assert abs(otf[..., 0, 0].real.item() - 1.0) < 1e-4


class TestDataFidelityFFT:
    def test_output_shape(self):
        """Data fidelity should preserve spatial dimensions."""
        y = torch.randn(1, 3, 64, 64)
        x0 = torch.randn(1, 3, 64, 64)
        kernel = torch.ones(1, 1, 9, 9) / 81.0
        otf = _psf_to_otf(kernel, (64, 64))
        rho = torch.tensor([1.0]).reshape(1, 1, 1, 1)
        result = _data_fidelity_fft(y, x0, otf, rho)
        assert result.shape == (1, 3, 64, 64)

    def test_identity_kernel(self):
        """With identity-like kernel and large rho, result should be close to x0."""
        y = torch.randn(1, 3, 64, 64)
        x0 = torch.randn(1, 3, 64, 64)
        # Nearly identity kernel (single pixel)
        kernel = torch.zeros(1, 1, 9, 9)
        kernel[..., 4, 4] = 1.0
        otf = _psf_to_otf(kernel, (64, 64))
        rho = torch.tensor([100.0]).reshape(1, 1, 1, 1)
        result = _data_fidelity_fft(y, x0, otf, rho)
        # With large rho, result should be close to x0
        mse = ((result - x0) ** 2).mean().item()
        assert mse < 0.1, f"MSE too large: {mse}"


class TestModelLoading:
    def test_weights_exist(self):
        """Check that the weights file exists."""
        weights_path = Path(__file__).resolve().parents[1] / "src" / "methods" / "diffpir" / "weights" / "ddpm_lbc.pt"
        assert weights_path.exists(), f"Weights not found at {weights_path}"

    def test_load_model(self):
        """Load the DDPM model and verify it returns expected structure."""
        weights_path = Path(__file__).resolve().parents[1] / "src" / "methods" / "diffpir" / "weights" / "ddpm_lbc.pt"
        if not weights_path.exists():
            return  # Skip if weights not available

        device = torch.device("cpu")
        model, schedule = _load_model(weights_path, device)

        assert "num_timesteps" in schedule
        assert schedule["num_timesteps"] == 1000
        assert "betas" in schedule
        assert "alphas_cumprod" in schedule
        assert schedule["betas"].shape[0] == 1000

    def test_model_forward(self):
        """Run a forward pass through the model (smoke test)."""
        weights_path = Path(__file__).resolve().parents[1] / "src" / "methods" / "diffpir" / "weights" / "ddpm_lbc.pt"
        if not weights_path.exists():
            return  # Skip if weights not available

        device = torch.device("cpu")
        model, _ = _load_model(weights_path, device)

        x = torch.randn(1, 3, 64, 64)
        t = torch.tensor([50])
        with torch.no_grad():
            pred = model(x, t)

        assert pred.shape == (1, 3, 64, 64), f"Expected (1,3,64,64), got {pred.shape}"
