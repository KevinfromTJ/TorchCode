"""Full BatchNorm1d module task."""

TASK = {
    "title": "BatchNorm Full Module",
    "difficulty": "Hard",
    "function_name": "MyBatchNorm1d",
    "hint": (
        "Implement BatchNorm1d as an nn.Module with train/eval behavior, running_mean, "
        "running_var, affine weight/bias, and momentum. Normalize over the batch dimension "
        "for inputs of shape (B, C)."
    ),
    "tests": [
        {
            "name": "Matches nn.BatchNorm1d in train mode",
            "code": "\n"
            "import torch, torch.nn as nn\n"
            "torch.manual_seed(0)\n"
            "x = torch.randn(8, 6)\n"
            "m1 = {fn}(6, eps=1e-5, momentum=0.1)\n"
            "m2 = nn.BatchNorm1d(6, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)\n"
            "m2.weight.data.copy_(m1.weight.data)\n"
            "m2.bias.data.copy_(m1.bias.data)\n"
            "m2.running_mean.data.copy_(m1.running_mean.data)\n"
            "m2.running_var.data.copy_(m1.running_var.data)\n"
            "m1.train(); m2.train()\n"
            "y1 = m1(x)\n"
            "y2 = m2(x)\n"
            "assert torch.allclose(y1, y2, atol=1e-5), f'Max diff: {(y1 - y2).abs().max():.6f}'\n"
        },
        {
            "name": "Running stats update in train mode",
            "code": "\n"
            "import torch\n"
            "m = {fn}(4)\n"
            "before_mean = m.running_mean.clone()\n"
            "before_var = m.running_var.clone()\n"
            "m.train()\n"
            "m(torch.randn(16, 4))\n"
            "assert not torch.allclose(m.running_mean, before_mean), 'running_mean should update in train mode'\n"
            "assert not torch.allclose(m.running_var, before_var), 'running_var should update in train mode'\n"
        },
        {
            "name": "Eval mode uses running stats",
            "code": "\n"
            "import torch\n"
            "m = {fn}(3)\n"
            "m.running_mean.copy_(torch.tensor([1.0, -2.0, 0.5]))\n"
            "m.running_var.copy_(torch.tensor([4.0, 9.0, 16.0]))\n"
            "m.eval()\n"
            "x = torch.tensor([[1.0, 1.0, 1.0], [3.0, 4.0, 5.0]])\n"
            "y = m(x)\n"
            "expected = (x - m.running_mean) / torch.sqrt(m.running_var + m.eps)\n"
            "expected = expected * m.weight + m.bias\n"
            "assert torch.allclose(y, expected, atol=1e-6), 'Eval mode should use running stats only'\n"
        },
        {
            "name": "Has buffers and parameters",
            "code": "\n"
            "m = {fn}(5)\n"
            "assert hasattr(m, 'running_mean') and hasattr(m, 'running_var')\n"
            "assert tuple(m.running_mean.shape) == (5,)\n"
            "assert tuple(m.running_var.shape) == (5,)\n"
            "assert tuple(m.weight.shape) == (5,)\n"
            "assert tuple(m.bias.shape) == (5,)\n"
        },
        {
            "name": "Gradient flow",
            "code": "\n"
            "import torch\n"
            "m = {fn}(4)\n"
            "x = torch.randn(10, 4, requires_grad=True)\n"
            "m.train()\n"
            "m(x).sum().backward()\n"
            "assert x.grad is not None, 'x.grad is None'\n"
            "assert m.weight.grad is not None and m.bias.grad is not None, 'Missing parameter gradients'\n"
        },
    ],
}
