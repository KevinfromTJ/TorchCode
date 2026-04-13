"""Binary Cross-Entropy with logits loss task."""

TASK = {
    "title": "Binary Cross-Entropy Loss",
    "difficulty": "Easy",
    "function_name": "bce_loss",
    "hint": (
        "Treat inputs as logits, not probabilities. Use the numerically stable "
        "BCE-with-logits form: max(x, 0) - x * y + log1p(exp(-abs(x))). "
        "Return the mean over all elements."
    ),
    "tests": [
        {
            "name": "Matches BCEWithLogitsLoss",
            "code": "\n"
            "import torch\n"
            "import torch.nn.functional as F\n"
            "torch.manual_seed(0)\n"
            "logits = torch.randn(16)\n"
            "targets = torch.randint(0, 2, (16,)).float()\n"
            "loss = {fn}(logits, targets)\n"
            "ref = F.binary_cross_entropy_with_logits(logits, targets)\n"
            "assert torch.allclose(loss, ref, atol=1e-5), f'{loss.item():.6f} vs {ref.item():.6f}'\n"
        },
        {
            "name": "Supports matrix inputs and soft targets",
            "code": "\n"
            "import torch\n"
            "import torch.nn.functional as F\n"
            "torch.manual_seed(1)\n"
            "logits = torch.randn(4, 5)\n"
            "targets = torch.rand(4, 5)\n"
            "loss = {fn}(logits, targets)\n"
            "ref = F.binary_cross_entropy_with_logits(logits, targets)\n"
            "assert loss.dim() == 0, 'Loss must be scalar'\n"
            "assert torch.allclose(loss, ref, atol=1e-5), 'Mismatch on matrix input'\n"
        },
        {
            "name": "Numerical stability on large logits",
            "code": "\n"
            "import torch\n"
            "logits = torch.tensor([1000.0, -1000.0, 80.0, -80.0])\n"
            "targets = torch.tensor([1.0, 0.0, 1.0, 0.0])\n"
            "loss = {fn}(logits, targets)\n"
            "assert torch.isfinite(loss), 'Loss should stay finite on large logits'\n"
            "assert loss.item() < 1e-4, f'Confident correct predictions should have tiny loss, got {loss.item():.6f}'\n"
        },
        {
            "name": "Wrong confident predictions incur large loss",
            "code": "\n"
            "import torch\n"
            "good = {fn}(torch.tensor([8.0, -8.0]), torch.tensor([1.0, 0.0]))\n"
            "bad = {fn}(torch.tensor([-8.0, 8.0]), torch.tensor([1.0, 0.0]))\n"
            "assert bad.item() > good.item() + 5.0, 'Bad predictions should have much larger loss'\n"
        },
        {
            "name": "Gradient flow",
            "code": "\n"
            "import torch\n"
            "logits = torch.randn(10, requires_grad=True)\n"
            "targets = torch.randint(0, 2, (10,)).float()\n"
            "{fn}(logits, targets).backward()\n"
            "assert logits.grad is not None, 'Missing gradients on logits'\n"
        },
    ],
}
