"""Full DAPO loss with asymmetric clipping and global token normalization."""

TASK = {
    "title": "DAPO Loss (Token-Level GRPO Variant)",
    "difficulty": "Hard",
    "function_name": "dapo_loss",
    "hint": (
        "Compute group-normalized advantages like GRPO, then build the token-level "
        "clipped objective. DAPO differs from full GRPO in two key ways: use a global "
        "token normalization over all active tokens in the batch, and support an "
        "asymmetric upper clipping bound such as 1 + clip_ratio_high."
    ),
    "tests": [
        {
            "name": "Basic shape & type",
            "code": "\n"
            "import torch\n"
            "from torch import Tensor\n"
            "torch.manual_seed(0)\n"
            "logps = torch.randn(4, 5, requires_grad=True)\n"
            "old_logps = torch.randn(4, 5)\n"
            "ref_logps = torch.randn(4, 5)\n"
            "rewards = torch.randn(4)\n"
            "group_ids = torch.tensor([0, 0, 1, 1])\n"
            "completion_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 0, 0, 0]], dtype=torch.float32)\n"
            "loss = {fn}(logps, old_logps, ref_logps, rewards, group_ids, completion_mask)\n"
            "assert isinstance(loss, Tensor) and loss.dim() == 0, 'Loss must be a scalar Tensor'\n"
        },
        {
            "name": "Numeric check vs reference",
            "code": "\n"
            "import torch\n"
            "from torch import Tensor\n"
            "\n"
            "def _reference_dapo_loss(logps: Tensor, old_logps: Tensor, ref_logps: Tensor, rewards: Tensor, group_ids: Tensor, completion_mask: Tensor, clip_ratio: float = 0.2, clip_ratio_high: float = 0.28, beta: float = 0.1, eps: float = 1e-5) -> Tensor:\n"
            "    advantages = torch.empty_like(rewards)\n"
            "    for gid in group_ids.unique():\n"
            "        mask = group_ids == gid\n"
            "        r_g = rewards[mask]\n"
            "        advantages[mask] = (r_g - r_g.mean()) / (r_g.std(unbiased=False) + eps)\n"
            "    advantages = advantages.detach().unsqueeze(1)\n"
            "    ratio = torch.exp(logps - old_logps.detach())\n"
            "    unclipped = ratio * advantages\n"
            "    clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio_high) * advantages\n"
            "    kl = torch.exp(ref_logps.detach() - logps) - (ref_logps.detach() - logps) - 1.0\n"
            "    token_objective = torch.minimum(unclipped, clipped) - beta * kl\n"
            "    return -(token_objective * completion_mask).sum() / completion_mask.sum().clamp_min(1.0)\n"
            "\n"
            "logps = torch.tensor([[-0.2, -0.1, -0.3], [-0.6, -0.5, -0.4], [-0.3, -0.8, -1.0], [-0.9, -1.1, -1.2]])\n"
            "old_logps = torch.tensor([[-0.3, -0.2, -0.4], [-0.4, -0.4, -0.4], [-0.5, -0.6, -0.9], [-0.7, -1.0, -1.3]])\n"
            "ref_logps = torch.tensor([[-0.25, -0.15, -0.35], [-0.55, -0.45, -0.45], [-0.35, -0.7, -0.95], [-0.8, -1.0, -1.1]])\n"
            "rewards = torch.tensor([1.0, 0.8, 0.3, 0.1])\n"
            "group_ids = torch.tensor([0, 0, 1, 1])\n"
            "completion_mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=torch.float32)\n"
            "loss_student = {fn}(logps, old_logps, ref_logps, rewards, group_ids, completion_mask, clip_ratio=0.2, clip_ratio_high=0.28, beta=0.1)\n"
            "loss_ref = _reference_dapo_loss(logps, old_logps, ref_logps, rewards, group_ids, completion_mask, clip_ratio=0.2, clip_ratio_high=0.28, beta=0.1)\n"
            "assert torch.allclose(loss_student, loss_ref, atol=1e-5, rtol=1e-5), f'{loss_student.item():.6f} vs {loss_ref.item():.6f}'\n"
        },
        {
            "name": "Zero group advantages leave only KL term",
            "code": "\n"
            "import torch\n"
            "logps = torch.tensor([[-0.2, -0.4], [-0.1, -0.5], [-1.0, -1.2], [-0.9, -1.3]], requires_grad=True)\n"
            "old_logps = logps.detach().clone() - 0.1\n"
            "ref_logps = torch.tensor([[-0.4, -0.3], [-0.3, -0.7], [-1.2, -1.0], [-1.1, -1.1]])\n"
            "rewards = torch.tensor([1.0, 1.0, 5.0, 5.0])\n"
            "group_ids = torch.tensor([0, 0, 1, 1])\n"
            "completion_mask = torch.ones_like(logps)\n"
            "loss = {fn}(logps, old_logps, ref_logps, rewards, group_ids, completion_mask, beta=0.3)\n"
            "kl = torch.exp(ref_logps - logps.detach()) - (ref_logps - logps.detach()) - 1.0\n"
            "expected = 0.3 * kl.mean()\n"
            "assert torch.allclose(loss.detach(), expected, atol=1e-5), 'When group advantages are zero, the loss should reduce to the KL penalty'\n"
        },
        {
            "name": "Masked tokens do not affect loss",
            "code": "\n"
            "import torch\n"
            "logps = torch.tensor([[-0.2, -0.3, -9.0], [-0.4, -0.5, 7.0]], requires_grad=True)\n"
            "old_logps = torch.tensor([[-0.1, -0.2, 3.0], [-0.5, -0.4, -8.0]])\n"
            "ref_logps = torch.tensor([[-0.2, -0.2, 1.0], [-0.5, -0.5, -6.0]])\n"
            "rewards = torch.tensor([1.0, 0.0])\n"
            "group_ids = torch.tensor([0, 0])\n"
            "completion_mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.float32)\n"
            "loss1 = {fn}(logps, old_logps, ref_logps, rewards, group_ids, completion_mask)\n"
            "logps2 = logps.detach().clone()\n"
            "logps2[:, 2] = torch.tensor([100.0, -100.0])\n"
            "loss2 = {fn}(logps2, old_logps, ref_logps, rewards, group_ids, completion_mask)\n"
            "assert torch.allclose(loss1.detach(), loss2, atol=1e-6), 'Masked tokens should not affect the loss'\n"
        },
        {
            "name": "Global token normalization weights longer sequences more",
            "code": "\n"
            "import torch\n"
            "ratio_hi = torch.log(torch.tensor(2.0))\n"
            "logps = torch.tensor([[ratio_hi, ratio_hi, ratio_hi], [0.0, 0.0, 0.0]], requires_grad=True)\n"
            "old_logps = torch.zeros_like(logps)\n"
            "ref_logps = torch.zeros_like(logps)\n"
            "rewards = torch.tensor([1.0, 0.0])\n"
            "group_ids = torch.tensor([0, 0])\n"
            "completion_mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.float32)\n"
            "loss = {fn}(logps, old_logps, ref_logps, rewards, group_ids, completion_mask, clip_ratio=10.0, clip_ratio_high=10.0, beta=0.0)\n"
            "adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-5)\n"
            "expected = -((3 * 2.0 * adv[0]) + (1 * 1.0 * adv[1])) / 4.0\n"
            "assert torch.allclose(loss.detach(), expected, atol=1e-6), 'DAPO should average over active tokens globally, not per sequence'\n"
        },
        {
            "name": "Gradient flows to current logps only",
            "code": "\n"
            "import torch\n"
            "torch.manual_seed(1)\n"
            "logps = torch.randn(4, 3, requires_grad=True)\n"
            "old_logps = torch.randn(4, 3, requires_grad=True)\n"
            "ref_logps = torch.randn(4, 3, requires_grad=True)\n"
            "rewards = torch.randn(4, requires_grad=True)\n"
            "group_ids = torch.tensor([0, 0, 1, 1])\n"
            "completion_mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=torch.float32)\n"
            "loss = {fn}(logps, old_logps, ref_logps, rewards, group_ids, completion_mask)\n"
            "loss.backward()\n"
            "assert logps.grad is not None, 'Gradients should flow through current logps'\n"
            "assert old_logps.grad is None, 'old_logps should be treated as fixed behavior policy'\n"
            "assert ref_logps.grad is None, 'ref_logps should be treated as fixed reference policy'\n"
            "assert rewards.grad is None, 'rewards / advantages should be detached from the graph'\n"
        },
    ],
}
