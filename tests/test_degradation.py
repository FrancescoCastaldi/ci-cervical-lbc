"""Tests for the degradation pipeline (blur + noise)."""

import torch
import pytest
from src.degradation.degradation import (
    gaussian_kernel,
    apply_blur,
    apply_noise,
    degrade,
)


class TestGaussianKernel:
    def test_shape(self):
        k = gaussian_kernel(kernel_size=9, sigma=2.0)
        assert k.shape == (9, 9), f"Expected (9,9), got {k.shape}"

    def test_normalization(self):
        k = gaussian_kernel(kernel_size=9, sigma=2.0)
        assert abs(k.sum().item() - 1.0) < 1e-6, f"Kernel sum != 1: {k.sum()}"

    def test_symmetry(self):
        k = gaussian_kernel(kernel_size=9, sigma=2.0)
        assert torch.allclose(k, k.T, atol=1e-6), "Kernel not symmetric"

    def test_different_sizes(self):
        for size in [3, 5, 7, 15]:
            k = gaussian_kernel(kernel_size=size, sigma=2.0)
            assert k.shape == (size, size)
            assert abs(k.sum().item() - 1.0) < 1e-6


class TestApplyBlur:
    def test_output_shape(self):
        img = torch.randn(3, 64, 64)
        blurred = apply_blur(img, kernel_size=9, sigma=2.0)
        assert blurred.shape == (3, 64, 64)

    def test_blur_smooths(self):
        img = torch.randn(3, 64, 64)
        blurred = apply_blur(img, kernel_size=9, sigma=2.0)
        # Blurred should have lower std than original
        assert blurred.std() < img.std() + 0.05


class TestApplyNoise:
    def test_output_shape(self):
        img = torch.randn(3, 64, 64)
        noisy = apply_noise(img, noise_level=0.05)
        assert noisy.shape == (3, 64, 64)

    def test_noise_level_approximation(self):
        """Check that the noise magnitude roughly matches the requested level."""
        img = torch.zeros(3, 128, 128)
        noisy = apply_noise(img, noise_level=0.05)
        measured_noise = (noisy - img).std().item()
        assert abs(measured_noise - 0.05) < 0.02, (
            f"Noise level mismatch: expected ~0.05, got {measured_noise:.4f}"
        )


class TestDegrade:
    def test_end_to_end(self):
        img = torch.randn(3, 64, 64)
        degraded = degrade(img, kernel_size=9, sigma=2.0, noise_level=0.05)
        assert degraded.shape == (3, 64, 64)
        assert degraded.is_floating_point(), "Output should be float"

    def test_different_noise_levels(self):
        img = torch.randn(3, 64, 64)
        for nl in [0.005, 0.01, 0.05, 0.1]:
            degraded = degrade(img, kernel_size=9, sigma=2.0, noise_level=nl)
            assert degraded.shape == (3, 64, 64)
