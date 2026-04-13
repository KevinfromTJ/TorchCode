"""Full GSPO loss with sequence-level importance sampling and masked token KL."""

TASK = {
    "title": "GSPO (Group Sequence Policy Optimization) Loss",
    "difficulty": "Hard",
    "function_name": "gspo_loss",
    "hint": (
        "First compute GRPO-style group-normalized advantages from rewards. Then build "
        "a sequence-level ratio by averaging token log-ratios over valid tokens only: "
        "seq_ratio = exp(sum(mask * (logps - old_logps)) / valid_tokens). Clip that "
        "ratio once per sequence, not per token. KL can still be token-level, then "
        "masked-average it per sequence before the batch mean."
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
            "def _reference_gspo_loss(logps: Tensor, old_logps: Tensor, ref_logps: Tensor, rewards: Tensor, group_ids: Tensor, completion_mask: Tensor, clip_ratio: float = 0.2, beta: float = 0.1, eps: float = 1e-5) -> Tensor:\n"
            "    advantages = torch.empty_like(rewards)\n"
            "    for gid in group_ids.unique():\n"
            "        mask = group_ids == gid\n"
            "        r_g = rewards[mask]\n"
            "        advantages[mask] = (r_g - r_g.mean()) / (r_g.std(unbiased=False) + eps)\n"
            "    advantages = advantages.detach()\n"
            "    valid = completion_mask.sum(dim=-1).clamp_min(1.0)\n"
            "    seq_log_ratio = ((logps - old_logps.detach()) * completion_mask).sum(dim=-1) / valid\n"
            "    seq_ratio = torch.exp(seq_log_ratio)\n"
            "    clipped = torch.clamp(seq_ratio, 1 - clip_ratio, 1 + clip_ratio)\n"
            "    seq_objective = torch.minimum(seq_ratio * advantages, clipped * advantages)\n"
            "    kl = torch.exp(ref_logps.detach() - logps) - (ref_logps.detach() - logps) - 1.0\n"
            "    seq_kl = (kl * completion_mask).sum(dim=-1) / valid\n"
            "    return -(seq_objective - beta * seq_kl).mean()\n"
            "\n"
            "logps = torch.tensor([[-0.2, -0.1, -0.3], [-0.7, -0.5, -0.4], [-0.3, -0.8, -1.0], [-0.9, -1.1, -1.2]])\n"
            "old_logps = torch.tensor([[-0.3, -0.2, -0.4], [-0.6, -0.6, -0.5], [-0.5, -0.6, -0.9], [-0.7, -1.0, -1.3]])\n"
            "ref_logps = torch.tensor([[-0.25, -0.15, -0.35], [-0.55, -0.45, -0.45], [-0.35, -0.7, -0.95], [-0.8, -1.0, -1.1]])\n"
            "rewards = torch.tensor([1.0, 0.8, 0.3, 0.1])\n"
            "group_ids = torch.tensor([0, 0, 1, 1])\n"
            "completion_mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=torch.float32)\n"
            "loss_student = {fn}(logps, old_logps, ref_logps, rewards, group_ids, completion_mask, clip_ratio=0.2, beta=0.1)\n"
            "loss_ref = _reference_gspo_loss(logps, old_logps, ref_logps, rewards, group_ids, completion_mask, clip_ratio=0.2, beta=0.1)\n"
            "assert torch.allclose(loss_student, loss_ref, atol=1e-5, rtol=1e-5), f'{loss_student.item():.6f} vs {loss_ref.item():.6f}'\n"
        },
        {
            "name": "Masked tokens do not affect sequence ratio",
            "code": "\n"
            "import torch\n"
            "logps = torch.tensor([[0.2, 0.2, -99.0], [0.0, 0.0, 99.0]], requires_grad=True)\n"
            "old_logps = torch.zeros_like(logps)\n"
            "ref_logps = torch.zeros_like(logps)\n"
            "rewards = torch.tensor([1.0, 0.0])\n"
            "group_ids = torch.tensor([0, 0])\n"
            "completion_mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.float32)\n"
            "loss1 = {fn}(logps, old_logps, ref_logps, rewards, group_ids, completion_mask, beta=0.0)\n"
            "logps2 = logps.detach().clone()\n"
            "logps2[:, 2] = torch.tensor([100.0, -100.0])\n"
            "loss2 = {fn}(logps2, old_logps, ref_logps, rewards, group_ids, completion_mask, beta=0.0)\n"
            "assert torch.allclose(loss1.detach(), loss2, atol=1e-6), 'Masked tokens should not affect GSPO sequence ratios'\n"
        },
        {
            "name": "Sequence ratio uses geometric mean over valid tokens",
            "code": "\n"
            "import torch\n"
            "logps = torch.tensor([[0.4, 0.4], [0.0, 0.0]], requires_grad=True)\n"
            "old_logps = torch.tensor([[0.0, 0.0], [0.0, 0.0]])\n"
            "ref_logps = torch.zeros_like(logps)\n"
            "rewards = torch.tensor([1.0, 0.0])\n"
            "group_ids = torch.tensor([0, 0])\n"
            "completion_mask = torch.ones_like(logps)\n"
            "loss = {fn}(logps, old_logps, ref_logps, rewards, group_ids, completion_mask, clip_ratio=10.0, beta=0.0)\n"
            "adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-5)\n"
            "seq_ratio = torch.exp(torch.tensor(0.4))\n"
            "expected = -torch.stack([seq_ratio * adv[0], torch.tensor(1.0) * adv[1]]).mean()\n"
            "assert torch.allclose(loss.detach(), expected, atol=1e-6), 'GSPO should use one ratio per sequence via the mean log-ratio'\n"
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
