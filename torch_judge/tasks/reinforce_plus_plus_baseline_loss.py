"""REINFORCE++ loss task for p > 1 with group baseline."""

TASK = {
    "title": "REINFORCE++ Loss (p > 1, with Baseline)",
    "difficulty": "Hard",
    "function_name": "reinforce_plus_plus_baseline_loss",
    "hint": (
        "For p>1, first subtract the per-group baseline: reward - mean(group rewards). "
        "Then globally normalize those centered rewards across the batch, detach them, "
        "broadcast to tokens, and return the negative masked average of advantage * logps."
    ),
    "tests": [
        {
            "name": "Basic shape & type",
            "code": "\n"
            "import torch\n"
            "from torch import Tensor\n"
            "logps = torch.randn(4, 5, requires_grad=True)\n"
            "rewards = torch.randn(4)\n"
            "group_ids = torch.tensor([0,0,1,1])\n"
            "completion_mask = torch.tensor([[1,1,1,1,1],[1,1,1,0,0],[1,1,1,1,0],[1,1,0,0,0]], dtype=torch.float32)\n"
            "loss = {fn}(logps, rewards, group_ids, completion_mask)\n"
            "assert isinstance(loss, Tensor) and loss.dim() == 0, 'Loss must be a scalar Tensor'\n"
        },
        {
            "name": "Numeric check vs reference",
            "code": "\n"
            "import torch\n"
            "from torch import Tensor\n"
            "def _ref(logps: Tensor, rewards: Tensor, group_ids: Tensor, completion_mask: Tensor, eps: float = 1e-5) -> Tensor:\n"
            "    centered = torch.empty_like(rewards)\n"
            "    for gid in group_ids.unique():\n"
            "        mask = group_ids == gid\n"
            "        r_g = rewards[mask]\n"
            "        centered[mask] = r_g - r_g.mean()\n"
            "    adv = centered / (centered.std(unbiased=False) + eps)\n"
            "    token_obj = logps * adv.detach().unsqueeze(1)\n"
            "    return -(token_obj * completion_mask).sum() / completion_mask.sum().clamp_min(1.0)\n"
            "logps = torch.tensor([[-0.2, -0.1, -0.3], [-0.6, -0.5, -0.4], [-0.3, -0.8, -1.0], [-0.9, -1.1, -1.2]])\n"
            "rewards = torch.tensor([1.0, 0.8, 0.3, 0.1])\n"
            "group_ids = torch.tensor([0,0,1,1])\n"
            "completion_mask = torch.tensor([[1,1,1],[1,1,0],[1,1,1],[1,0,0]], dtype=torch.float32)\n"
            "loss = {fn}(logps, rewards, group_ids, completion_mask)\n"
            "ref = _ref(logps, rewards, group_ids, completion_mask)\n"
            "assert torch.allclose(loss, ref, atol=1e-5), f'{loss.item():.6f} vs {ref.item():.6f}'\n"
        },
        {
            "name": "Equal rewards within group give zero advantage",
            "code": "\n"
            "import torch\n"
            "logps = torch.tensor([[-0.2, -0.4], [-0.1, -0.5], [-1.0, -1.2], [-0.9, -1.3]], requires_grad=True)\n"
            "rewards = torch.tensor([1.0, 1.0, 5.0, 5.0])\n"
            "group_ids = torch.tensor([0,0,1,1])\n"
            "completion_mask = torch.ones_like(logps)\n"
            "loss = {fn}(logps, rewards, group_ids, completion_mask)\n"
            "assert torch.allclose(loss.detach(), torch.tensor(0.0), atol=1e-6), 'If all rewards equal within each group, baseline-centered advantages should vanish'\n"
        },
        {
            "name": "Masked tokens do not affect loss",
            "code": "\n"
            "import torch\n"
            "logps = torch.tensor([[-0.2, -0.3, 99.0], [-0.4, -0.5, -99.0]], requires_grad=True)\n"
            "rewards = torch.tensor([1.0, 0.0])\n"
            "group_ids = torch.tensor([0,0])\n"
            "completion_mask = torch.tensor([[1,1,0],[1,1,0]], dtype=torch.float32)\n"
            "loss1 = {fn}(logps, rewards, group_ids, completion_mask)\n"
            "logps2 = logps.detach().clone(); logps2[:, 2] = torch.tensor([-100.0, 100.0])\n"
            "loss2 = {fn}(logps2, rewards, group_ids, completion_mask)\n"
            "assert torch.allclose(loss1.detach(), loss2, atol=1e-6), 'Masked tokens should not affect the loss'\n"
        },
        {
            "name": "Gradient flows to logps only",
            "code": "\n"
            "import torch\n"
            "logps = torch.randn(4, 3, requires_grad=True)\n"
            "rewards = torch.randn(4, requires_grad=True)\n"
            "group_ids = torch.tensor([0,0,1,1])\n"
            "completion_mask = torch.tensor([[1,1,1],[1,1,0],[1,1,1],[1,0,0]], dtype=torch.float32)\n"
            "loss = {fn}(logps, rewards, group_ids, completion_mask)\n"
            "loss.backward()\n"
            "assert logps.grad is not None, 'Gradients should flow through logps'\n"
            "assert rewards.grad is None, 'Rewards should be detached from the graph'\n"
        },
    ],
}
