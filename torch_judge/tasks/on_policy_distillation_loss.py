"""On-policy distillation loss task with masked token-level KL."""

TASK = {
    "title": "On-Policy Distillation Loss",
    "difficulty": "Hard",
    "function_name": "on_policy_distillation_loss",
    "hint": (
        "Treat the sampled trajectory tokens as coming from the current student policy. Distill a frozen teacher onto those same on-policy tokens: "
        "compute token-level KL between teacher and student distributions at temperature T, mask invalid tokens, and normalize over active tokens only."
    ),
    "tests": [
        {
            "name": "Scalar output",
            "code": "\n"
            "import torch\n"
            "student = torch.randn(2, 4, 6, requires_grad=True)\n"
            "teacher = torch.randn(2, 4, 6)\n"
            "mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.float32)\n"
            "loss = {fn}(student, teacher, mask)\n"
            "assert loss.dim() == 0, 'Loss must be scalar'\n"
        },
        {
            "name": "Numeric check vs reference",
            "code": "\n"
            "import torch\n"
            "import torch.nn.functional as F\n"
            "student = torch.tensor([[[2.0, 0.0, -1.0], [0.1, 0.2, 0.3]], [[0.5, -0.1, 0.0], [1.0, 0.0, -1.0]]], requires_grad=True)\n"
            "teacher = torch.tensor([[[3.0, -1.0, 0.0], [0.0, 0.5, 1.0]], [[0.0, 0.0, 0.0], [1.2, -0.3, -0.9]]])\n"
            "mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.float32)\n"
            "temperature = 2.0\n"
            "loss = {fn}(student, teacher, mask, temperature=temperature)\n"
            "student_log_probs = F.log_softmax(student / temperature, dim=-1)\n"
            "teacher_probs = F.softmax(teacher / temperature, dim=-1)\n"
            "token_kl = (teacher_probs * (torch.log(teacher_probs.clamp_min(1e-12)) - student_log_probs)).sum(dim=-1) * (temperature ** 2)\n"
            "ref = (token_kl * mask).sum() / mask.sum()\n"
            "assert torch.allclose(loss, ref, atol=1e-5), f'{loss.item():.6f} vs {ref.item():.6f}'\n"
        },
        {
            "name": "Masked tokens do not affect loss",
            "code": "\n"
            "import torch\n"
            "student = torch.tensor([[[0.0, 1.0], [2.0, 3.0], [100.0, -100.0]]], requires_grad=True)\n"
            "teacher = torch.tensor([[[1.0, 0.0], [3.0, 2.0], [-100.0, 100.0]]])\n"
            "mask = torch.tensor([[1, 1, 0]], dtype=torch.float32)\n"
            "loss1 = {fn}(student, teacher, mask)\n"
            "student2 = student.detach().clone()\n"
            "teacher2 = teacher.detach().clone()\n"
            "student2[:, 2] = torch.tensor([-999.0, 999.0])\n"
            "teacher2[:, 2] = torch.tensor([999.0, -999.0])\n"
            "loss2 = {fn}(student2, teacher2, mask)\n"
            "assert torch.allclose(loss1.detach(), loss2, atol=1e-6), 'Masked tokens should not affect the on-policy distillation loss'\n"
        },
        {
            "name": "Student matching teacher gives lower loss",
            "code": "\n"
            "import torch\n"
            "teacher = torch.tensor([[[2.0, -1.0, 0.5], [0.1, 0.2, 0.3]]])\n"
            "mask = torch.ones(1, 2)\n"
            "good = {fn}(teacher.clone().requires_grad_(True), teacher, mask)\n"
            "bad = {fn}(torch.tensor([[[-1.0, 2.0, 0.0], [1.0, -2.0, 0.0]]], requires_grad=True), teacher, mask)\n"
            "assert good.item() < bad.item(), 'Matching teacher logits should reduce distillation loss'\n"
        },
        {
            "name": "Gradient flows to student only",
            "code": "\n"
            "import torch\n"
            "student = torch.randn(2, 3, 5, requires_grad=True)\n"
            "teacher = torch.randn(2, 3, 5, requires_grad=True)\n"
            "mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.float32)\n"
            "loss = {fn}(student, teacher, mask)\n"
            "loss.backward()\n"
            "assert student.grad is not None, 'student should receive gradients'\n"
            "assert teacher.grad is None, 'teacher should be treated as fixed'\n"
        },
    ],
}
