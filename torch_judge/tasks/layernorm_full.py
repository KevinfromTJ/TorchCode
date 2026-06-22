"""Full LayerNorm module task."""

TASK = {
    "title": "LayerNorm Full Module",
    "difficulty": "Medium",
    "function_name": "MyLayerNorm",
    "hint": (
        "Implement an nn.Module version of LayerNorm with learnable weight and bias. "
        "Normalize over the last dimension only: y = (x - mean) / sqrt(var + eps) * weight + bias."
    ),
    "tests": [
        {
            "name": "Matches nn.LayerNorm",
            "code": "\n"
            "import torch, torch.nn as nn\n"
            "torch.manual_seed(0)\n"
            "x = torch.randn(2, 4, 8)\n"
            "m1 = {fn}(8, eps=1e-5)\n"
            "m2 = nn.LayerNorm(8, eps=1e-5)\n"
            "m2.weight.data.copy_(m1.weight.data)\n"
            "m2.bias.data.copy_(m1.bias.data)\n"
            "y1 = m1(x)\n"
            "y2 = m2(x)\n"
            "assert torch.allclose(y1, y2, atol=1e-5), f'Max diff: {(y1 - y2).abs().max():.6f}'\n"
        },
        {
            "name": "Has parameters",
            "code": "\n"
            "m = {fn}(16)\n"
            "assert hasattr(m, 'weight'), 'Need self.weight'\n"
            "assert hasattr(m, 'bias'), 'Need self.bias'\n"
            "assert tuple(m.weight.shape) == (16,)\n"
            "assert tuple(m.bias.shape) == (16,)\n"
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
            "name": "Per-token mean approximately zero with default params",
            "code": "\n"
            "import torch\n"
            "m = {fn}(10)\n"
            "x = torch.randn(4, 6, 10)\n"
            "y = m(x)\n"
            "assert torch.allclose(y.mean(dim=-1), torch.zeros_like(y.mean(dim=-1)), atol=1e-5), 'Last-dim mean should be ~0'\n"
        },
        {
            "name": "Gradient flow",
            "code": "\n"
            "import torch\n"
            "m = {fn}(8)\n"
            "x = torch.randn(2, 3, 8, requires_grad=True)\n"
            "m(x).sum().backward()\n"
            "assert x.grad is not None, 'x.grad is None'\n"
            "assert m.weight.grad is not None and m.bias.grad is not None, 'Missing parameter gradients'\n"
        },
    ],
}
