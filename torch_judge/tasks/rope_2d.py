"""2D Rotary Position Embedding task for vision patch grids."""

TASK = {
    "title": "2D Rotary Position Embedding (2D RoPE)",
    "difficulty": "Hard",
    "function_name": "apply_2d_rope",
    "hint": (
        "Assume q and k are flattened image patches with shape (B, H*W, D) and D divisible by 4. "
        "Split the feature dimension into two halves: the first half uses row positions, the second "
        "half uses column positions. Apply standard 1D RoPE separately to each half, then concatenate."
    ),
    "tests": [
        {
            "name": "Output shapes",
            "code": "\n"
            "import torch\n"
            "q = torch.randn(2, 16, 64)\n"
            "k = torch.randn(2, 16, 64)\n"
            "q_rot, k_rot = {fn}(q, k, height=4, width=4)\n"
            "assert q_rot.shape == q.shape, f'Q shape: {q_rot.shape}'\n"
            "assert k_rot.shape == k.shape, f'K shape: {k_rot.shape}'\n"
        },
        {
            "name": "Preserves norm",
            "code": "\n"
            "import torch\n"
            "torch.manual_seed(0)\n"
            "q = torch.randn(1, 9, 32)\n"
            "k = torch.randn(1, 9, 32)\n"
            "q_rot, k_rot = {fn}(q, k, height=3, width=3)\n"
            "assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-4), '2D RoPE should preserve query norms'\n"
            "assert torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-4), '2D RoPE should preserve key norms'\n"
        },
        {
            "name": "Top-left patch stays unchanged",
            "code": "\n"
            "import torch\n"
            "torch.manual_seed(0)\n"
            "q = torch.randn(1, 9, 16)\n"
            "k = torch.randn(1, 9, 16)\n"
            "q_rot, k_rot = {fn}(q, k, height=3, width=3)\n"
            "assert torch.allclose(q_rot[:, 0], q[:, 0], atol=1e-6), 'Position (0,0) should have zero rotation for q'\n"
            "assert torch.allclose(k_rot[:, 0], k[:, 0], atol=1e-6), 'Position (0,0) should have zero rotation for k'\n"
        },
        {
            "name": "Row rotation is shared within the same row",
            "code": "\n"
            "import torch\n"
            "height, width, dim = 2, 3, 8\n"
            "q = torch.zeros(1, height * width, dim)\n"
            "k = torch.zeros(1, height * width, dim)\n"
            "q[:, :, :4] = torch.tensor([1.0, 2.0, 3.0, 4.0])\n"
            "k[:, :, :4] = torch.tensor([1.0, 2.0, 3.0, 4.0])\n"
            "q_rot, _ = {fn}(q, k, height=height, width=width)\n"
            "assert torch.allclose(q_rot[0, 0, :4], q_rot[0, 1, :4], atol=1e-6), 'Same row should share the same row-axis rotation'\n"
            "assert not torch.allclose(q_rot[0, 0, :4], q_rot[0, 3, :4], atol=1e-4), 'Different rows should rotate the row half differently'\n"
        },
        {
            "name": "Column rotation is shared within the same column",
            "code": "\n"
            "import torch\n"
            "height, width, dim = 2, 3, 8\n"
            "q = torch.zeros(1, height * width, dim)\n"
            "k = torch.zeros(1, height * width, dim)\n"
            "q[:, :, 4:] = torch.tensor([1.0, 2.0, 3.0, 4.0])\n"
            "k[:, :, 4:] = torch.tensor([1.0, 2.0, 3.0, 4.0])\n"
            "q_rot, _ = {fn}(q, k, height=height, width=width)\n"
            "assert torch.allclose(q_rot[0, 0, 4:], q_rot[0, 3, 4:], atol=1e-6), 'Same column should share the same column-axis rotation'\n"
            "assert not torch.allclose(q_rot[0, 0, 4:], q_rot[0, 1, 4:], atol=1e-4), 'Different columns should rotate the column half differently'\n"
        },
        {
            "name": "Gradient flow",
            "code": "\n"
            "import torch\n"
            "q = torch.randn(1, 9, 16, requires_grad=True)\n"
            "k = torch.randn(1, 9, 16, requires_grad=True)\n"
            "q_rot, k_rot = {fn}(q, k, height=3, width=3)\n"
            "(q_rot.sum() + k_rot.sum()).backward()\n"
            "assert q.grad is not None and k.grad is not None, 'Missing gradients'\n"
        },
    ],
}
