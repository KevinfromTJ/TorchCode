"""Full GDPO loss with decoupled multi-reward normalization."""

TASK = {
    "title": "GDPO (Decoupled Multi-Reward GRPO) Loss",
    "difficulty": "Hard",
    "function_name": "gdpo_loss",
    "hint": (
        "Assume rewards has shape (B, R). For each reward dimension, normalize within "
        "each group independently. Then weight-sum the normalized rewards across reward "
        "dimensions, and apply one more batch-level normalization to the aggregated "
        "advantage. Use that final detached advantage in a masked GRPO-style clipped loss."
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
            "rewards = torch.randn(4, 3)\n"
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
            "def _reference_gdpo_loss(logps: Tensor, old_logps: Tensor, ref_logps: Tensor, rewards: Tensor, group_ids: Tensor, completion_mask: Tensor, reward_weights: Tensor, clip_ratio: float = 0.2, beta: float = 0.1, eps: float = 1e-5) -> Tensor:\n"
            "    if rewards.dim() == 1:\n"
            "        rewards = rewards.unsqueeze(-1)\n"
            "    per_reward_adv = torch.empty_like(rewards)\n"
            "    for ridx in range(rewards.shape[-1]):\n"
            "        for gid in group_ids.unique():\n"
            "            mask = group_ids == gid\n"
            "            r_g = rewards[mask, ridx]\n"
            "            per_reward_adv[mask, ridx] = (r_g - r_g.mean()) / (r_g.std(unbiased=False) + eps)\n"
            "    agg = (per_reward_adv * reward_weights.view(1, -1)).sum(dim=-1)\n"
            "    agg = (agg - agg.mean()) / (agg.std(unbiased=False) + eps)\n"
            "    agg = agg.detach().unsqueeze(1)\n"
            "    ratio = torch.exp(logps - old_logps.detach())\n"
            "    unclipped = ratio * agg\n"
            "    clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * agg\n"
            "    kl = torch.exp(ref_logps.detach() - logps) - (ref_logps.detach() - logps) - 1.0\n"
            "    token_objective = torch.minimum(unclipped, clipped) - beta * kl\n"
            "    valid = completion_mask.sum(dim=-1).clamp_min(1.0)\n"
            "    seq_objective = (token_objective * completion_mask).sum(dim=-1) / valid\n"
            "    return -seq_objective.mean()\n"
            "\n"
            "logps = torch.tensor([[-0.2, -0.1, -0.3], [-0.6, -0.5, -0.4], [-0.3, -0.8, -1.0], [-0.9, -1.1, -1.2]])\n"
            "old_logps = torch.tensor([[-0.3, -0.2, -0.4], [-0.4, -0.4, -0.4], [-0.5, -0.6, -0.9], [-0.7, -1.0, -1.3]])\n"
            "ref_logps = torch.tensor([[-0.25, -0.15, -0.35], [-0.55, -0.45, -0.45], [-0.35, -0.7, -0.95], [-0.8, -1.0, -1.1]])\n"
            "rewards = torch.tensor([[1.0, 0.3], [0.8, 0.1], [0.3, 0.9], [0.1, 0.2]])\n"
            "reward_weights = torch.tensor([0.7, 0.3])\n"
            "group_ids = torch.tensor([0, 0, 1, 1])\n"
            "completion_mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=torch.float32)\n"
            "loss_student = {fn}(logps, old_logps, ref_logps, rewards, group_ids, completion_mask, reward_weights=reward_weights, clip_ratio=0.2, beta=0.1)\n"
            "loss_ref = _reference_gdpo_loss(logps, old_logps, ref_logps, rewards, group_ids, completion_mask, reward_weights=reward_weights, clip_ratio=0.2, beta=0.1)\n"
            "assert torch.allclose(loss_student, loss_ref, atol=1e-5, rtol=1e-5), f'{loss_student.item():.6f} vs {loss_ref.item():.6f}'\n"
        },
        {
            "name": "Per-reward scaling invariance",
            "code": "\n"
            "import torch\n"
            "torch.manual_seed(0)\n"
            "logps = torch.randn(4, 3)\n"
            "old_logps = torch.randn(4, 3)\n"
            "ref_logps = torch.randn(4, 3)\n"
            "rewards = torch.tensor([[1.0, 2.0], [0.0, 1.0], [2.0, -1.0], [1.0, -2.0]])\n"
            "group_ids = torch.tensor([0, 0, 1, 1])\n"
            "completion_mask = torch.ones(4, 3)\n"
            "loss1 = {fn}(logps, old_logps, ref_logps, rewards, group_ids, completion_mask, beta=0.0)\n"
            "rewards_scaled = rewards.clone()\n"
            "rewards_scaled[:, 1] = rewards_scaled[:, 1] * 100.0\n"
            "loss2 = {fn}(logps, old_logps, ref_logps, rewards_scaled, group_ids, completion_mask, beta=0.0)\n"
            "assert torch.allclose(loss1, loss2, atol=1e-5), 'Each reward dimension should be normalized independently before aggregation'\n"
        },
        {
            "name": "Masked tokens do not affect loss",
            "code": "\n"
            "import torch\n"
            "logps = torch.tensor([[-0.2, -0.3, -9.0], [-0.4, -0.5, 7.0], [-0.1, -0.2, 8.0], [-0.5, -0.7, -6.0]], requires_grad=True)\n"
            "old_logps = torch.tensor([[-0.1, -0.2, 3.0], [-0.5, -0.4, -8.0], [-0.3, -0.4, 1.0], [-0.6, -0.8, -2.0]])\n"
            "ref_logps = torch.tensor([[-0.2, -0.2, 1.0], [-0.5, -0.5, -6.0], [-0.2, -0.3, 5.0], [-0.5, -0.6, 4.0]])\n"
            "rewards = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.2], [0.1, -0.2]])\n"
            "group_ids = torch.tensor([0, 0, 1, 1])\n"
            "completion_mask = torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]], dtype=torch.float32)\n"
            "loss1 = {fn}(logps, old_logps, ref_logps, rewards, group_ids, completion_mask)\n"
            "logps2 = logps.detach().clone()\n"
            "logps2[:, 2] = torch.tensor([100.0, -100.0, 50.0, -50.0])\n"
            "loss2 = {fn}(logps2, old_logps, ref_logps, rewards, group_ids, completion_mask)\n"
            "assert torch.allclose(loss1.detach(), loss2, atol=1e-6), 'Masked tokens should not affect the loss'\n"
        },
        {
            "name": "Gradient flows to current logps only",
            "code": "\n"
            "import torch\n"
            "torch.manual_seed(1)\n"
            "logps = torch.randn(4, 3, requires_grad=True)\n"
            "old_logps = torch.randn(4, 3, requires_grad=True)\n"
            "ref_logps = torch.randn(4, 3, requires_grad=True)\n"
            "rewards = torch.randn(4, 2, requires_grad=True)\n"
            "group_ids = torch.tensor([0, 0, 1, 1])\n"
            "completion_mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=torch.float32)\n"
            "loss = {fn}(logps, old_logps, ref_logps, rewards, group_ids, completion_mask)\n"
            "loss.backward()\n"
            "assert logps.grad is not None, 'Gradients should flow through current logps'\n"
            "assert old_logps.grad is None, 'old_logps should be treated as fixed behavior policy'\n"
            "assert ref_logps.grad is None, 'ref_logps should be treated as fixed reference policy'\n"
            "assert rewards.grad is None, 'reward-derived advantages should be detached from the graph'\n"
        },
    ],
}
