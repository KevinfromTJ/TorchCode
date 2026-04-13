"""Symmetric InfoNCE / CLIP-style contrastive loss task."""

TASK = {
    "title": "InfoNCE Contrastive Loss",
    "difficulty": "Hard",
    "function_name": "infonce_loss",
    "hint": (
        "Use in-batch negatives. Normalize both embedding batches along the last "
        "dimension, compute logits = z1 @ z2.T / temperature, and use labels = "
        "arange(B). A modern CLIP-style version averages cross-entropy in both "
        "directions: rows and columns."
    ),
    "tests": [
        {
            "name": "Scalar output",
            "code": "\n"
            "import torch\n"
            "torch.manual_seed(0)\n"
            "z1 = torch.randn(6, 8)\n"
            "z2 = torch.randn(6, 8)\n"
            "loss = {fn}(z1, z2)\n"
            "assert loss.dim() == 0, 'Loss must be scalar'\n"
        },
        {
            "name": "Matches symmetric reference formula",
            "code": "\n"
            "import torch\n"
            "import torch.nn.functional as F\n"
            "torch.manual_seed(1)\n"
            "z1 = torch.randn(5, 7)\n"
            "z2 = torch.randn(5, 7)\n"
            "temperature = 0.2\n"
            "loss = {fn}(z1, z2, temperature=temperature)\n"
            "a = F.normalize(z1, dim=-1)\n"
            "b = F.normalize(z2, dim=-1)\n"
            "logits = a @ b.T / temperature\n"
            "labels = torch.arange(z1.shape[0])\n"
            "ref = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))\n"
            "assert torch.allclose(loss, ref, atol=1e-5), f'{loss.item():.6f} vs {ref.item():.6f}'\n"
        },
        {
            "name": "Aligned pairs beat shuffled pairs",
            "code": "\n"
            "import torch\n"
            "z = torch.eye(4)\n"
            "good = {fn}(z, z, temperature=0.07)\n"
            "bad = {fn}(z, z[torch.tensor([1, 0, 3, 2])], temperature=0.07)\n"
            "assert good.item() < 1e-2, f'Aligned pairs should have near-zero loss, got {good.item():.6f}'\n"
            "assert bad.item() > good.item() + 1.0, 'Shuffled positives should be much worse'\n"
        },
        {
            "name": "Scale invariant after normalization",
            "code": "\n"
            "import torch\n"
            "torch.manual_seed(2)\n"
            "z1 = torch.randn(6, 10)\n"
            "z2 = torch.randn(6, 10)\n"
            "loss1 = {fn}(z1, z2, temperature=0.1)\n"
            "loss2 = {fn}(z1 * 7.0, z2 * 0.25, temperature=0.1)\n"
            "assert torch.allclose(loss1, loss2, atol=1e-5), 'L2 normalization should make the loss scale invariant'\n"
        },
        {
            "name": "Gradient flow",
            "code": "\n"
            "import torch\n"
            "torch.manual_seed(3)\n"
            "z1 = torch.randn(8, 16, requires_grad=True)\n"
            "z2 = torch.randn(8, 16, requires_grad=True)\n"
            "{fn}(z1, z2, temperature=0.07).backward()\n"
            "assert z1.grad is not None and z2.grad is not None, 'Missing gradients on embeddings'\n"
        },
    ],
}
