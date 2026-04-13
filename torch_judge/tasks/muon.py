"""Muon optimizer task (simplified Newton-Schulz orthogonalized momentum)."""

TASK = {
    "title": "Muon Optimizer (Simplified)",
    "difficulty": "Hard",
    "function_name": "MyMuon",
    "hint": (
        "Maintain momentum m = beta*m + (1-beta)*(grad + wd*param). For matrix-like "
        "tensors (ndim >= 2), orthogonalize the momentum with Newton-Schulz: normalize, "
        "then iterate X = 1.5X - 0.5 X (X^T X). For vectors, use momentum directly. "
        "Update: p -= lr * update. Implement zero_grad() as well."
    ),
    "tests": [
        {
            "name": "Parameters change after step",
            "code": "\n"
            "import torch\n"
            "torch.manual_seed(0)\n"
            "w = torch.randn(8, 4, requires_grad=True)\n"
            "opt = {fn}([w], lr=0.05)\n"
            "(w ** 2).sum().backward()\n"
            "before = w.detach().clone()\n"
            "opt.step()\n"
            "assert not torch.equal(w.detach(), before), 'Parameters should change after step()'\n"
        },
        {
            "name": "Matrix update matches reference orthogonalization",
            "code": "\n"
            "import torch\n"
            "\n"
            "def _orthogonalize(g: torch.Tensor, ns_steps: int, eps: float) -> torch.Tensor:\n"
            "    orig_shape = g.shape\n"
            "    x = g\n"
            "    if x.ndim > 2:\n"
            "        x = x.reshape(x.shape[0], -1)\n"
            "    transposed = False\n"
            "    if x.shape[0] > x.shape[1]:\n"
            "        x = x.T\n"
            "        transposed = True\n"
            "    x = x / (x.norm() + eps)\n"
            "    for _ in range(ns_steps):\n"
            "        x = 1.5 * x - 0.5 * (x @ (x.T @ x))\n"
            "    if transposed:\n"
            "        x = x.T\n"
            "    return x.reshape(orig_shape)\n"
            "\n"
            "torch.manual_seed(1)\n"
            "w = torch.randn(6, 4, requires_grad=True)\n"
            "grad = torch.randn_like(w)\n"
            "w.grad = grad.clone()\n"
            "lr = 0.2\n"
            "ns_steps = 4\n"
            "eps = 1e-8\n"
            "opt = {fn}([w], lr=lr, beta=0.0, weight_decay=0.0, ns_steps=ns_steps, eps=eps)\n"
            "before = w.detach().clone()\n"
            "opt.step()\n"
            "expected = before - lr * _orthogonalize(grad, ns_steps=ns_steps, eps=eps)\n"
            "assert torch.allclose(w.detach(), expected, atol=1e-5, rtol=1e-5), 'Matrix update mismatch vs reference'\n"
        },
        {
            "name": "Vector parameter skips orthogonalization",
            "code": "\n"
            "import torch\n"
            "p = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)\n"
            "p.grad = torch.tensor([0.3, -0.2, 0.1])\n"
            "opt = {fn}([p], lr=0.5, beta=0.0, weight_decay=0.0)\n"
            "before = p.detach().clone()\n"
            "opt.step()\n"
            "expected = before - 0.5 * torch.tensor([0.3, -0.2, 0.1])\n"
            "assert torch.allclose(p.detach(), expected, atol=1e-6), '1D params should use raw momentum update'\n"
        },
        {
            "name": "Weight decay is applied",
            "code": "\n"
            "import torch\n"
            "p = torch.tensor([1.0, -1.0], requires_grad=True)\n"
            "p.grad = torch.tensor([0.5, 0.5])\n"
            "lr = 0.1\n"
            "wd = 0.2\n"
            "opt = {fn}([p], lr=lr, beta=0.0, weight_decay=wd)\n"
            "before = p.detach().clone()\n"
            "opt.step()\n"
            "expected = before - lr * (torch.tensor([0.5, 0.5]) + wd * before)\n"
            "assert torch.allclose(p.detach(), expected, atol=1e-6), 'Weight decay should be included in update direction'\n"
        },
        {
            "name": "zero_grad works",
            "code": "\n"
            "import torch\n"
            "w = torch.randn(4, 3, requires_grad=True)\n"
            "opt = {fn}([w], lr=0.01)\n"
            "(w ** 2).sum().backward()\n"
            "assert w.grad is not None and w.grad.abs().sum() > 0\n"
            "opt.zero_grad()\n"
            "assert w.grad is not None and w.grad.abs().sum() == 0, 'zero_grad should zero all gradients'\n"
        },
    ],
}
