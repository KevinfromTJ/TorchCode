"""Classic knowledge distillation loss task."""

TASK = {
    "title": "Knowledge Distillation (KD) Loss",
    "difficulty": "Medium",
    "function_name": "kd_loss",
    "hint": (
        "Compute hard-label CE on student logits, plus a soft distillation KL term between "
        "teacher and student distributions at temperature T. Standard Hinton-style KD uses "
        "KL(log_softmax(student/T), softmax(teacher/T)) * T^2. Then mix: alpha * KD + (1-alpha) * CE."
    ),
    "tests": [
        {
            "name": "Scalar output",
            "code": "\n"
            "import torch\n"
            "student = torch.randn(4, 6, requires_grad=True)\n"
            "teacher = torch.randn(4, 6)\n"
            "targets = torch.randint(0, 6, (4,))\n"
            "loss = {fn}(student, teacher, targets)\n"
            "assert loss.dim() == 0, 'Loss must be scalar'\n"
        },
        {
            "name": "Numeric check vs reference",
            "code": "\n"
            "import torch\n"
            "import torch.nn.functional as F\n"
            "student = torch.tensor([[2.0, 0.0, -1.0], [0.1, 0.2, 0.3]])\n"
            "teacher = torch.tensor([[3.0, -1.0, 0.0], [0.0, 0.5, 1.0]])\n"
            "targets = torch.tensor([0, 2])\n"
            "temperature = 2.0\n"
            "alpha = 0.7\n"
            "loss = {fn}(student, teacher, targets, temperature=temperature, alpha=alpha)\n"
            "kd = F.kl_div(F.log_softmax(student / temperature, dim=-1), F.softmax(teacher / temperature, dim=-1), reduction='batchmean') * (temperature ** 2)\n"
            "ce = F.cross_entropy(student, targets)\n"
            "ref = alpha * kd + (1 - alpha) * ce\n"
            "assert torch.allclose(loss, ref, atol=1e-5), f'{loss.item():.6f} vs {ref.item():.6f}'\n"
        },
        {
            "name": "Teacher match gives lower KD term",
            "code": "\n"
            "import torch\n"
            "teacher = torch.tensor([[2.0, -1.0, 0.5]])\n"
            "targets = torch.tensor([0])\n"
            "good = {fn}(teacher.clone(), teacher, targets, alpha=1.0)\n"
            "bad = {fn}(torch.tensor([[-1.0, 2.0, 0.0]]), teacher, targets, alpha=1.0)\n"
            "assert good.item() < bad.item(), 'Matching teacher logits should reduce KD loss'\n"
        },
        {
            "name": "alpha zero reduces to cross-entropy",
            "code": "\n"
            "import torch\n"
            "import torch.nn.functional as F\n"
            "torch.manual_seed(0)\n"
            "student = torch.randn(5, 4)\n"
            "teacher = torch.randn(5, 4)\n"
            "targets = torch.randint(0, 4, (5,))\n"
            "loss = {fn}(student, teacher, targets, alpha=0.0)\n"
            "ref = F.cross_entropy(student, targets)\n"
            "assert torch.allclose(loss, ref, atol=1e-6), 'alpha=0 should equal standard CE'\n"
        },
        {
            "name": "Gradient flows to student only",
            "code": "\n"
            "import torch\n"
            "student = torch.randn(6, 5, requires_grad=True)\n"
            "teacher = torch.randn(6, 5, requires_grad=True)\n"
            "targets = torch.randint(0, 5, (6,))\n"
            "loss = {fn}(student, teacher, targets)\n"
            "loss.backward()\n"
            "assert student.grad is not None, 'student.grad is None'\n"
            "assert teacher.grad is None, 'teacher should be treated as fixed'\n"
        },
    ],
}
