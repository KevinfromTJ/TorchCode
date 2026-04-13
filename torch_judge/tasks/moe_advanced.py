"""Advanced MoE task (top-k routing + auxiliary load-balancing loss)."""

TASK = {
    "title": "MoE (Advanced: Top-k + Aux Loss)",
    "difficulty": "Hard",
    "function_name": "AdvancedMoE",
    "hint": (
        "Flatten tokens to N x D. Router logits -> top-k experts per token. Normalize "
        "top-k logits with softmax and combine selected expert outputs. Also compute a "
        "Switch-style load balancing loss: importance = softmax(logits).mean(0), "
        "load = top1 assignment frequency, aux = num_experts * sum(importance * load)."
    ),
    "tests": [
        {
            "name": "Output and aux loss shape",
            "code": "\n"
            "import torch, torch.nn as nn\n"
            "moe = {fn}(d_model=32, d_ff=64, num_experts=4, top_k=2)\n"
            "assert isinstance(moe, nn.Module)\n"
            "x = torch.randn(2, 8, 32)\n"
            "out, aux = moe(x, return_aux_loss=True)\n"
            "assert out.shape == (2, 8, 32), f'Output shape: {out.shape}'\n"
            "assert aux.dim() == 0, 'Aux loss must be scalar'\n"
            "out_only = moe(x)\n"
            "assert out_only.shape == (2, 8, 32), 'forward(x) should return only output tensor'\n"
        },
        {
            "name": "Top-1 routing picks selected expert",
            "code": "\n"
            "import torch\n"
            "import torch.nn as nn\n"
            "\n"
            "moe = {fn}(d_model=4, d_ff=8, num_experts=3, top_k=1)\n"
            "for e in range(3):\n"
            "    lin = nn.Linear(4, 4, bias=False)\n"
            "    lin.weight.data = torch.eye(4) * float(e + 1)\n"
            "    moe.experts[e] = lin\n"
            "\n"
            "class FixedRouter(nn.Module):\n"
            "    def forward(self, x):\n"
            "        logits = torch.tensor([[[10.0, 0.0, 0.0], [0.0, 0.0, 10.0]]], dtype=x.dtype, device=x.device)\n"
            "        return logits.expand(x.shape[0], x.shape[1], -1)\n"
            "\n"
            "moe.router = FixedRouter()\n"
            "x = torch.tensor([[[1.0, -2.0, 0.5, 3.0], [0.2, 0.1, -0.4, 0.7]]])\n"
            "out, _ = moe(x, return_aux_loss=True)\n"
            "expected = torch.stack([x[:, 0, :], 3.0 * x[:, 1, :]], dim=1)\n"
            "assert torch.allclose(out, expected, atol=1e-5), 'Top-1 routing output mismatch'\n"
        },
        {
            "name": "Aux loss matches reference formula",
            "code": "\n"
            "import torch\n"
            "import torch.nn as nn\n"
            "\n"
            "def _ref_aux(logits: torch.Tensor) -> torch.Tensor:\n"
            "    probs = torch.softmax(logits, dim=-1)\n"
            "    importance = probs.mean(dim=(0, 1))\n"
            "    top1 = logits.argmax(dim=-1).reshape(-1)\n"
            "    load = torch.bincount(top1, minlength=logits.shape[-1]).float() / top1.numel()\n"
            "    return logits.shape[-1] * torch.sum(importance * load)\n"
            "\n"
            "moe = {fn}(d_model=6, d_ff=12, num_experts=3, top_k=2)\n"
            "fixed_logits = torch.tensor([\n"
            "    [[4.0, 0.0, -1.0], [3.0, 1.0, 0.0]],\n"
            "    [[0.0, 4.0, 1.0], [-1.0, 0.0, 4.0]],\n"
            "])\n"
            "\n"
            "class FixedRouter(nn.Module):\n"
            "    def forward(self, x):\n"
            "        return fixed_logits.to(dtype=x.dtype, device=x.device)\n"
            "\n"
            "moe.router = FixedRouter()\n"
            "x = torch.randn(2, 2, 6)\n"
            "_, aux = moe(x, return_aux_loss=True)\n"
            "expected = _ref_aux(fixed_logits)\n"
            "assert torch.allclose(aux, expected, atol=1e-6), f'{aux.item():.6f} vs {expected.item():.6f}'\n"
        },
        {
            "name": "Collapsed routing has larger aux loss than balanced routing",
            "code": "\n"
            "import torch\n"
            "import torch.nn as nn\n"
            "\n"
            "x = torch.randn(2, 3, 8)\n"
            "balanced_logits = torch.tensor([\n"
            "    [[6.0, 0.0], [0.0, 6.0], [6.0, 0.0]],\n"
            "    [[0.0, 6.0], [6.0, 0.0], [0.0, 6.0]],\n"
            "])\n"
            "collapsed_logits = torch.tensor([\n"
            "    [[6.0, 0.0], [6.0, 0.0], [6.0, 0.0]],\n"
            "    [[6.0, 0.0], [6.0, 0.0], [6.0, 0.0]],\n"
            "])\n"
            "\n"
            "moe = {fn}(d_model=8, d_ff=16, num_experts=2, top_k=1)\n"
            "\n"
            "class RouterBalanced(nn.Module):\n"
            "    def forward(self, h):\n"
            "        return balanced_logits.to(dtype=h.dtype, device=h.device)\n"
            "\n"
            "class RouterCollapsed(nn.Module):\n"
            "    def forward(self, h):\n"
            "        return collapsed_logits.to(dtype=h.dtype, device=h.device)\n"
            "\n"
            "moe.router = RouterBalanced()\n"
            "_, aux_balanced = moe(x, return_aux_loss=True)\n"
            "moe.router = RouterCollapsed()\n"
            "_, aux_collapsed = moe(x, return_aux_loss=True)\n"
            "assert aux_collapsed > aux_balanced + 0.1, 'Collapsed routing should have noticeably larger aux loss'\n"
        },
        {
            "name": "Gradient flow with aux objective",
            "code": "\n"
            "import torch\n"
            "moe = {fn}(d_model=16, d_ff=32, num_experts=4, top_k=2)\n"
            "x = torch.randn(2, 4, 16, requires_grad=True)\n"
            "out, aux = moe(x, return_aux_loss=True)\n"
            "loss = out.mean() + 0.05 * aux\n"
            "loss.backward()\n"
            "assert x.grad is not None, 'x.grad is None'\n"
            "assert moe.router.weight.grad is not None, 'Router should receive gradients'\n"
        },
    ],
}
