# tests/ — Unit Tests

34 unit tests verifying the correctness of the implementations.

## Exam Utility

- **Code reliability**: each method has specific tests that verify its operation
- **Reproducibility**: the tests ensure implementations are numerically stable and correct
- **Coverage**: degradation (10), metrics (9), diffpir (7), unet (8), tv (8)

## Files

| File | Tests | What it verifies |
|---|---|---|
| `test_degradation.py` | 10 | Gaussian kernel, blur, AWGN, complete pipeline |
| `test_metrics.py` | 9 | PSNR, SSIM, edge cases (identical, black, different images) |
| `test_diffpir.py` | 7 | FFT data-fidelity, DDIM sampling, output shape |
| `test_unet.py` | 8 | Forward pass, gradients, output range, noise conditioning |
| `test_tv.py` | 8 | TV loss, kernel, blur, output range, convergence |

## Running

```bash
# All tests
python -m pytest tests/ -v

# Single file
python -m pytest tests/test_tv.py -v

# Single test
python -m pytest tests/test_tv.py::test_tv_loss -v

# With detailed output
python -m pytest tests/ -v --tb=short
```

## Test Structure

Tests use `pytest` with independent `test_*` functions (no classes). Each test creates synthetic inputs with `torch.randn` and verifies the output with `torch.allclose`.

**Concrete example** (from `test_degradation.py`):

```python
def test_gaussian_kernel_sum():
    kernel = gaussian_kernel(kernel_size=9, sigma=2.0)
    assert torch.allclose(kernel.sum(), torch.tensor(1.0), atol=1e-6)
    assert kernel.shape == (1, 1, 9, 9)
```

**Metrics example** (from `test_metrics.py`):

```python
def test_psnr_identical():
    x = torch.randn(1, 3, 64, 64)
    psnr = compute_psnr(x, x)
    assert psnr == float("inf")  # identical images → infinite PSNR
```

## What is Tested per Module

| Module | Specific coverage |
|---|---|
| **degradation** | kernel normalization, blur preserves mean, correct AWGN σ, reproducible end-to-end pipeline with seed |
| **metrics** | Infinite PSNR for identical inputs, PSNR = 0 for opposite images, SSIM = 1 for identical, SSIM < 1 for different, [0,1] and [-1,1] ranges |
| **diffpir** | FFT data-fidelity term, DDIM sampling preserves shape and [-1,1] range, zeta=0 matches pure denoising |
| **unet** | Forward pass with and without noise conditioning, gradients flow, output in [-1,1], shape preserved, skip connections active |
| **tv** | TV loss always ≥ 0, normalized Gaussian kernel, blur reduces variance, output in [-1,1], iterative convergence |

## Adding New Tests

1. Create `tests/test_<module>.py`
2. Import the functions to test
3. Write `test_*` functions without a class
4. Use `torch.allclose()` with `atol=1e-4` for numerical tolerance
5. Run with `python -m pytest tests/test_<module>.py -v`

## Dependencies

- `pytest` — test runner
- `torch` — tensors and operations
- `numpy` — additional metrics
- `scikit-image` — reference SSIM for cross-validation
