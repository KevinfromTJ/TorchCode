"""Self-distillation loss task with main/auxiliary heads."""

TASK = {
    "title": "Self-Distillation Loss",
    "difficulty": "Medium",
    "function_name": "self_distillation_loss",
    "hint": (
        "Use the main head as the teacher signal, detached from the graph. Compute a hard CE term on the main logits, "
        "plus a KD term from main -> auxiliary logits at temperature T. Return ce_weight * CE + distill_weight * KD."
    ),
    "tests": [
        {
            "name": "Scalar output",
            "code": "\n"
            "import torch\n"
            "main = torch.randn(4, 6, requires_grad=True)\n"
            "aux = torch.randn(4, 6, requires_grad=True)\n"
            "targets = torch.randint(0, 6, (4,))\n"
            "loss = {fn}(main, aux, targets)\n"
            "assert loss.dim() == 0, 'Loss must be scalar'\n"
        },
        {
            "name": "Numeric check vs reference",
            "code": "\n"
            "import torch\n"
            "import torch.nn.functional as F\n"
            "main = torch.tensor([[2.0, 0.0, -1.0], [0.1, 0.2, 0.3]], requires_grad=True)\n"
            "aux = torch.tensor([[1.5, -0.2, -0.5], [0.0, 0.4, 0.2]], requires_grad=True)\n"
            "targets = torch.tensor([0, 2])\n"
            "temperature = 2.0\n"
            "ce_weight = 1.0\n"
            "distill_weight = 0.5\n"
            "loss = {fn}(main, aux, targets, temperature=temperature, ce_weight=ce_weight, distill_weight=distill_weight)\n"
            "ce = F.cross_entropy(main, targets)\n"
            "kd = F.kl_div(F.log_softmax(aux / temperature, dim=-1), F.softmax(main.detach() / temperature, dim=-1), reduction='batchmean') * (temperature ** 2)\n"
            "ref = ce_weight * ce + distill_weight * kd\n"
            "assert torch.allclose(loss, ref, atol=1e-5), f'{loss.item():.6f} vs {ref.item():.6f}'\n"
        },
        {
            "name": "Aux matching main lowers distillation term",
            "code": "\n"
            "import torch\n"
            "main = torch.tensor([[2.0, -1.0, 0.5]])\n"
            "targets = torch.tensor([0])\n"
            "good = {fn}(main.clone().requires_grad_(True), main.clone().requires_grad_(True), targets, ce_weight=0.0, distill_weight=1.0)\n"
            "bad_aux = torch.tensor([[-1.0, 2.0, 0.0]], requires_grad=True)\n"
            "bad = {fn}(main.clone().requires_grad_(True), bad_aux, targets, ce_weight=0.0, distill_weight=1.0)\n"
            "assert good.item() < bad.item(), 'Aux head closer to main head should reduce self-distillation loss'\n"
        },
        {
            "name": "distill_weight zero reduces to CE on main head",
            "code": "\n"
            "import torch\n"
            "import torch.nn.functional as F\n"
            "torch.manual_seed(0)\n"
            "main = torch.randn(5, 4)\n"
            "aux = torch.randn(5, 4)\n"
            "targets = torch.randint(0, 4, (5,))\n"
            "loss = {fn}(main, aux, targets, distill_weight=0.0)\n"
            "ref = F.cross_entropy(main, targets)\n"
            "assert torch.allclose(loss, ref, atol=1e-6), 'distill_weight=0 should equal CE(main, targets)'\n"
        },
        {
            "name": "Gradient flows to both heads except teacher path detached",
            "code": "\n"
            "import torch\n"
            "main = torch.randn(6, 5, requires_grad=True)\n"
            "aux = torch.randn(6, 5, requires_grad=True)\n"
            "targets = torch.randint(0, 5, (6,))\n"
            "loss = {fn}(main, aux, targets)\n"
            "loss.backward()\n"
            "assert main.grad is not None, 'main head should receive CE gradients'\n"
            "assert aux.grad is not None, 'aux head should receive KD gradients'\n"
        },
    ],
}
