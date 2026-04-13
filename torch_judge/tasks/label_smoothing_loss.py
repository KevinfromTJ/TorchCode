"""Label smoothing cross-entropy loss task."""

TASK = {
    "title": "Label Smoothing Loss",
    "difficulty": "Medium",
    "function_name": "label_smoothing_loss",
    "hint": (
        "Compute log_probs = logits.log_softmax(dim=-1). For C classes, PyTorch-style "
        "label smoothing mixes the one-hot target with a uniform distribution: "
        "target_prob = (1 - s) * one_hot + s / C. Return the mean cross-entropy."
    ),
    "tests": [
        {
            "name": "Matches CrossEntropyLoss with label_smoothing",
            "code": "\n"
            "import torch\n"
            "import torch.nn.functional as F\n"
            "torch.manual_seed(0)\n"
            "logits = torch.randn(8, 6)\n"
            "targets = torch.randint(0, 6, (8,))\n"
            "smoothing = 0.1\n"
            "loss = {fn}(logits, targets, smoothing=smoothing)\n"
            "ref = F.cross_entropy(logits, targets, label_smoothing=smoothing)\n"
            "assert torch.allclose(loss, ref, atol=1e-5), f'{loss.item():.6f} vs {ref.item():.6f}'\n"
        },
        {
            "name": "Smoothing zero reduces to cross-entropy",
            "code": "\n"
            "import torch\n"
            "import torch.nn.functional as F\n"
            "torch.manual_seed(1)\n"
            "logits = torch.randn(5, 4)\n"
            "targets = torch.randint(0, 4, (5,))\n"
            "loss = {fn}(logits, targets, smoothing=0.0)\n"
            "ref = F.cross_entropy(logits, targets)\n"
            "assert torch.allclose(loss, ref, atol=1e-6), 'smoothing=0 should equal standard CE'\n"
        },
        {
            "name": "Uniform logits give log(C)",
            "code": "\n"
            "import torch, math\n"
            "logits = torch.zeros(3, 5)\n"
            "targets = torch.tensor([0, 2, 4])\n"
            "loss = {fn}(logits, targets, smoothing=0.2)\n"
            "expected = torch.tensor(math.log(5.0))\n"
            "assert torch.allclose(loss, expected, atol=1e-6), f'{loss.item():.6f} vs {expected.item():.6f}'\n"
        },
        {
            "name": "Smoothing penalizes overconfidence",
            "code": "\n"
            "import torch\n"
            "plain = {fn}(torch.tensor([[12.0, -12.0, -12.0]]), torch.tensor([0]), smoothing=0.0)\n"
            "smooth = {fn}(torch.tensor([[12.0, -12.0, -12.0]]), torch.tensor([0]), smoothing=0.1)\n"
            "assert smooth.item() > plain.item(), 'Label smoothing should increase loss on overconfident predictions'\n"
        },
        {
            "name": "Gradient flow",
            "code": "\n"
            "import torch\n"
            "logits = torch.randn(7, 3, requires_grad=True)\n"
            "targets = torch.randint(0, 3, (7,))\n"
            "{fn}(logits, targets, smoothing=0.1).backward()\n"
            "assert logits.grad is not None, 'Missing gradients on logits'\n"
        },
    ],
}
