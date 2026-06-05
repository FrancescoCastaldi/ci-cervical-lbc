"""Tests for evaluation metrics (PSNR, SSIM)."""

import torch
from src.eval.metrics import compute_psnr, compute_ssim, evaluate, _to_01


class TestPSNR:
    def test_identical_images(self):
        """PSNR should be very high for identical images.
        compute_psnr expects pred in [0,1] and gt in [-1,1] (dataset format)."""
        arr = torch.rand(3, 64, 64)
        pred = arr.clone()  # [0, 1]
        gt = arr * 2 - 1     # same content, but in [-1, 1]
        psnr_val = compute_psnr(pred, gt)
        assert psnr_val > 50.0, f"PSNR for identical images too low: {psnr_val:.2f}"

    def test_different_images(self):
        """PSNR should be lower for different images."""
        pred = torch.zeros(3, 64, 64)
        gt = (torch.ones(3, 64, 64) * 0.5) * 2 - 1  # [-1, 1]
        psnr_val = compute_psnr(pred, gt)
        assert psnr_val < 30.0

    def test_output_range(self):
        """PSNR should be a finite positive float."""
        pred = torch.rand(3, 64, 64)
        gt = torch.rand(3, 64, 64)
        psnr_val = compute_psnr(pred, gt)
        assert psnr_val > 0
        assert psnr_val < 100


class TestSSIM:
    def test_identical_images(self):
        """SSIM should be 1.0 for identical images.
        compute_ssim expects pred in [0,1] and gt in [-1,1] (dataset format)."""
        arr = torch.rand(3, 64, 64)
        pred = arr.clone()  # [0, 1]
        gt = arr * 2 - 1     # same content, but in [-1, 1]
        ssim_val = compute_ssim(pred, gt)
        assert abs(ssim_val - 1.0) < 1e-4, f"SSIM != 1.0: {ssim_val}"

    def test_output_range(self):
        """SSIM should be between -1 and 1."""
        pred = torch.rand(3, 64, 64)
        gt = torch.rand(3, 64, 64)
        ssim_val = compute_ssim(pred, gt)
        assert -1.0 <= ssim_val <= 1.0


class TestEvaluate:
    def test_returns_dict(self):
        """evaluate should return a dict with psnr and ssim keys."""
        pred = torch.rand(3, 64, 64)
        gt = torch.rand(3, 64, 64)
        result = evaluate(pred, gt)
        assert "psnr" in result
        assert "ssim" in result

    def test_gt_normalization(self):
        """Test that GT in [-1,1] is properly converted to [0,1]."""
        # GT in [-1,1] as from dataset, pred in [0,1] as from restoration
        gt = (torch.rand(3, 64, 64) * 2 - 1)  # [-1, 1]
        pred = torch.rand(3, 64, 64)  # [0, 1]
        result = evaluate(pred, gt)
        assert result["psnr"] > 0


class TestTo01:
    def test_conversion(self):
        t = torch.tensor([-1.0, 0.0, 1.0])
        result = _to_01(t)
        expected = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(result, expected), f"Got {result}, expected {expected}"

    def test_range(self):
        # _to_01 applies (t+1)/2 without clamping.
        # With input values at [-1, 0, 1], output must be [0, 0.5, 1].
        t = torch.tensor([-1.0, 0.0, 1.0])
        result = _to_01(t)
        expected = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(result, expected)
        # For a tensor already in [-1, 1], result must be in [0, 1].
        t2 = torch.rand(3, 64, 64) * 2 - 1
        result2 = _to_01(t2)
        assert result2.min() >= 0.0
        assert result2.max() <= 1.0
