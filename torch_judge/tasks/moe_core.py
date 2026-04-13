"""Core simplified MoE task (dense routing over all experts)."""

TASK = {
    "title": "MoE Core (Simplified Dense Routing)",
    "difficulty": "Hard",
    "function_name": "SimpleMoECore",
    "hint": (
        "Use router logits -> softmax to get expert weights per token. Run all experts "
        "on the same input token, stack their outputs, then compute the weighted sum "
        "across experts. This is a dense MoE core: no top-k sparsity yet."
    ),
    "tests": [
        {
            "name": "Output shape",
            "code": "\n"
            "import torch, torch.nn as nn\n"
            "moe = {fn}(d_model=16, num_experts=4)\n"
            "assert isinstance(moe, nn.Module)\n"
            "out = moe(torch.randn(2, 8, 16))\n"
            "assert out.shape == (2, 8, 16), f'Shape: {out.shape}'\n"
        },
        {
            "name": "Has router and experts",
            "code": "\n"
            "import torch\n"
            "moe = {fn}(d_model=8, num_experts=3)\n"
            "assert hasattr(moe, 'router'), 'Need self.router'\n"
            "assert hasattr(moe, 'experts'), 'Need self.experts'\n"
            "assert len(moe.experts) == 3, f'Expected 3 experts, got {len(moe.experts)}'\n"
        },
        {
            "name": "Weighted sum matches manual reference",
            "code": "\n"
            "import torch\n"
            "import torch.nn as nn\n"
            "\n"
            "moe = {fn}(d_model=4, num_experts=3)\n"
            "for e in range(3):\n"
            "    lin = nn.Linear(4, 4, bias=False)\n"
            "    lin.weight.data = torch.eye(4) * float(e + 1)\n"
            "    moe.experts[e] = lin\n"
            "\n"
            "class FixedRouter(nn.Module):\n"
            "    def forward(self, x):\n"
            "        logits = torch.tensor([[[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]]], dtype=x.dtype, device=x.device)\n"
            "        return logits.expand(x.shape[0], x.shape[1], -1)\n"
            "\n"
            "moe.router = FixedRouter()\n"
            "x = torch.tensor([[[1.0, 2.0, -1.0, 0.5], [0.2, -0.3, 0.4, 1.0]]])\n"
            "out = moe(x)\n"
            "\n"
            "logits = torch.tensor([[[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]]], dtype=x.dtype)\n"
            "weights = torch.softmax(logits, dim=-1)\n"
            "expert_stack = torch.stack([x, 2.0 * x, 3.0 * x], dim=2)\n"
            "expected = (weights.unsqueeze(-1) * expert_stack).sum(dim=2)\n"
            "assert torch.allclose(out, expected, atol=1e-5), 'Dense MoE weighted sum mismatch'\n"
        },
        {
            "name": "Gradient flow",
            "code": "\n"
            "import torch\n"
            "moe = {fn}(d_model=12, num_experts=4)\n"
            "x = torch.randn(2, 5, 12, requires_grad=True)\n"
            "loss = moe(x).pow(2).mean()\n"
            "loss.backward()\n"
            "assert x.grad is not None, 'x.grad is None'\n"
            "router_grads = [p.grad for p in moe.router.parameters()]\n"
            "expert_grads = [p.grad for ex in moe.experts for p in ex.parameters()]\n"
            "assert any(g is not None for g in router_grads), 'Router should receive gradients'\n"
            "assert any(g is not None for g in expert_grads), 'Experts should receive gradients'\n"
        },
    ],
}
