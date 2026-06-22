"""REINFORCE++ loss task for p = 1."""

TASK = {
    "title": "REINFORCE++ Loss (p = 1)",
    "difficulty": "Hard",
    "function_name": "reinforce_plus_plus_loss",
    "hint": (
        "For p=1, there is no per-prompt baseline because each prompt has only one sample. "
        "Compute globally normalized advantages from rewards over the batch, broadcast them "
        "across valid tokens, and return the negative masked average of advantage * logps."
    ),
    "tests": [
        {
            "name": "Basic shape & type",
            "code": "\n"
            "import torch\n"
            "from torch import Tensor\n"
            "logps = torch.randn(4, 5, requires_grad=True)\n"
            "rewards = torch.randn(4)\n"
            "completion_mask = torch.tensor([[1,1,1,1,1],[1,1,1,0,0],[1,1,1,1,0],[1,1,0,0,0]], dtype=torch.float32)\n"
            "loss = {fn}(logps, rewards, completion_mask)\n"
            "assert isinstance(loss, Tensor) and loss.dim() == 0, 'Loss must be a scalar Tensor'\n"
        },
        {
            "name": "Numeric check vs reference",
            "code": "\n"
            "import torch\n"
            "from torch import Tensor\n"
            "def _ref(logps: Tensor, rewards: Tensor, completion_mask: Tensor, eps: float = 1e-5) -> Tensor:\n"
            "    adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + eps)\n"
            "    token_obj = logps * adv.detach().unsqueeze(1)\n"
            "    return -(token_obj * completion_mask).sum() / completion_mask.sum().clamp_min(1.0)\n"
            "logps = torch.tensor([[-0.2, -0.1, -0.3], [-0.6, -0.5, -0.4], [-0.3, -0.8, -1.0], [-0.9, -1.1, -1.2]])\n"
            "rewards = torch.tensor([1.0, 0.8, 0.3, 0.1])\n"
            "completion_mask = torch.tensor([[1,1,1],[1,1,0],[1,1,1],[1,0,0]], dtype=torch.float32)\n"
            "loss = {fn}(logps, rewards, completion_mask)\n"
            "ref = _ref(logps, rewards, completion_mask)\n"
            "assert torch.allclose(loss, ref, atol=1e-5), f'{loss.item():.6f} vs {ref.item():.6f}'\n"
        },
        {
            "name": "Masked tokens do not affect loss",
            "code": "\n"
            "import torch\n"
            "logps = torch.tensor([[-0.2, -0.3, 99.0], [-0.4, -0.5, -99.0]], requires_grad=True)\n"
            "rewards = torch.tensor([1.0, 0.0])\n"
            "completion_mask = torch.tensor([[1,1,0],[1,1,0]], dtype=torch.float32)\n"
            "loss1 = {fn}(logps, rewards, completion_mask)\n"
            "logps2 = logps.detach().clone(); logps2[:, 2] = torch.tensor([-100.0, 100.0])\n"
            "loss2 = {fn}(logps2, rewards, completion_mask)\n"
            "assert torch.allclose(loss1.detach(), loss2, atol=1e-6), 'Masked tokens should not affect the loss'\n"
        },
        {
            "name": "Gradient flows to logps only",
            "code": "\n"
            "import torch\n"
            "logps = torch.randn(4, 3, requires_grad=True)\n"
            "rewards = torch.randn(4, requires_grad=True)\n"
            "completion_mask = torch.tensor([[1,1,1],[1,1,0],[1,1,1],[1,0,0]], dtype=torch.float32)\n"
            "loss = {fn}(logps, rewards, completion_mask)\n"
            "loss.backward()\n"
            "assert logps.grad is not None, 'Gradients should flow through logps'\n"
            "assert rewards.grad is None, 'Rewards should be detached from the graph'\n"
        },
    ],
}
