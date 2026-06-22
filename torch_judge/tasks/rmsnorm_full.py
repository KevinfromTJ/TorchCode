"""Full RMSNorm module task."""

TASK = {
    "title": "RMSNorm Full Module",
    "difficulty": "Medium",
    "function_name": "MyRMSNorm",
    "hint": (
        "RMSNorm normalizes using only the root-mean-square: y = x / sqrt(mean(x^2) + eps) * weight. "
        "No bias term is needed in the standard form. Normalize over the last dimension."
    ),
    "tests": [
        {
            "name": "Matches reference formula",
            "code": "\n"
            "import torch\n"
            "torch.manual_seed(0)\n"
            "x = torch.randn(2, 4, 8)\n"
            "m = {fn}(8, eps=1e-5)\n"
            "y = m(x)\n"
            "ref = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * m.weight\n"
            "assert torch.allclose(y, ref, atol=1e-5), f'Max diff: {(y - ref).abs().max():.6f}'\n"
        },
        {
            "name": "Has learnable weight",
            "code": "\n"
            "m = {fn}(16)\n"
            "assert hasattr(m, 'weight'), 'Need self.weight'\n"
            "assert tuple(m.weight.shape) == (16,)\n"
        },
        {
            "name": "Output shape preserved",
            "code": "\n"
            "import torch\n"
            "m = {fn}(12)\n"
            "x = torch.randn(3, 5, 12)\n"
            "y = m(x)\n"
            "assert y.shape == x.shape, f'Shape mismatch: {y.shape}'\n"
        },
        {
            "name": "RMS becomes one before scaling",
            "code": "\n"
            "import torch\n"
            "m = {fn}(10)\n"
            "x = torch.randn(4, 6, 10)\n"
            "y = m(x) / m.weight\n"
            "rms = torch.sqrt(y.pow(2).mean(dim=-1))\n"
            "assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5), 'Normalized RMS should be ~1'\n"
        },
        {
            "name": "Gradient flow",
            "code": "\n"
            "import torch\n"
            "m = {fn}(8)\n"
            "x = torch.randn(2, 3, 8, requires_grad=True)\n"
            "m(x).sum().backward()\n"
            "assert x.grad is not None, 'x.grad is None'\n"
            "assert m.weight.grad is not None, 'Missing weight gradient'\n"
        },
    ],
}
