"""Binary focal loss task."""

TASK = {
    "title": "Focal Loss",
    "difficulty": "Medium",
    "function_name": "focal_loss",
    "hint": (
        "Start from BCE-with-logits per element. Let p = sigmoid(logits), "
        "p_t = p for positive labels and 1-p for negative labels. Then "
        "loss = (1 - p_t)^gamma * BCE. If alpha is provided, weight positives "
        "by alpha and negatives by 1-alpha. Return the mean."
    ),
    "tests": [
        {
            "name": "Matches manual formula",
            "code": "\n"
            "import torch\n"
            "import torch.nn.functional as F\n"
            "torch.manual_seed(0)\n"
            "logits = torch.randn(12)\n"
            "targets = torch.randint(0, 2, (12,)).float()\n"
            "gamma = 2.0\n"
            "loss = {fn}(logits, targets, gamma=gamma)\n"
            "bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')\n"
            "p = torch.sigmoid(logits)\n"
            "p_t = p * targets + (1 - p) * (1 - targets)\n"
            "ref = ((1 - p_t) ** gamma * bce).mean()\n"
            "assert torch.allclose(loss, ref, atol=1e-5), f'{loss.item():.6f} vs {ref.item():.6f}'\n"
        },
        {
            "name": "Gamma zero reduces to BCE-with-logits",
            "code": "\n"
            "import torch\n"
            "import torch.nn.functional as F\n"
            "torch.manual_seed(1)\n"
            "logits = torch.randn(9)\n"
            "targets = torch.randint(0, 2, (9,)).float()\n"
            "loss = {fn}(logits, targets, gamma=0.0)\n"
            "ref = F.binary_cross_entropy_with_logits(logits, targets)\n"
            "assert torch.allclose(loss, ref, atol=1e-6), 'gamma=0 should equal BCE-with-logits'\n"
        },
        {
            "name": "Hard examples get larger weight",
            "code": "\n"
            "import torch\n"
            "easy = {fn}(torch.tensor([8.0, -8.0]), torch.tensor([1.0, 0.0]), gamma=2.0)\n"
            "hard = {fn}(torch.tensor([-8.0, 8.0]), torch.tensor([1.0, 0.0]), gamma=2.0)\n"
            "assert hard.item() > easy.item() + 5.0, 'Hard misclassified examples should produce much larger focal loss'\n"
        },
        {
            "name": "Alpha weighting is applied",
            "code": "\n"
            "import torch\n"
            "logits = torch.tensor([0.0, 0.0])\n"
            "targets = torch.tensor([1.0, 0.0])\n"
            "unweighted = {fn}(logits, targets, gamma=2.0, alpha=None)\n"
            "weighted = {fn}(logits, targets, gamma=2.0, alpha=0.9)\n"
            "assert not torch.allclose(unweighted, weighted), 'alpha should change the loss when class weights differ'\n"
        },
        {
            "name": "Gradient flow",
            "code": "\n"
            "import torch\n"
            "logits = torch.randn(11, requires_grad=True)\n"
            "targets = torch.randint(0, 2, (11,)).float()\n"
            "{fn}(logits, targets, gamma=2.0, alpha=0.25).backward()\n"
            "assert logits.grad is not None, 'Missing gradients on logits'\n"
        },
    ],
}
