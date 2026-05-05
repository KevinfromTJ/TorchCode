"""Full simplified Vision Transformer task."""

TASK = {
    "title": "Vision Transformer (ViT) Full Model",
    "difficulty": "Hard",
    "function_name": "VisionTransformer",
    "hint": (
        "Patchify the image, project patches to embeddings, prepend a learnable cls token, "
        "add a learnable position embedding, run several Transformer blocks, then classify "
        "from the final cls token. You can simplify details by using nn.Conv2d for patch "
        "embedding and nn.MultiheadAttention inside each block."
    ),
    "tests": [
        {
            "name": "Logits shape",
            "code": "\n"
            "import torch, torch.nn as nn\n"
            "torch.manual_seed(0)\n"
            "model = {fn}(img_size=32, patch_size=8, in_channels=3, num_classes=10, embed_dim=64, depth=2, num_heads=4)\n"
            "assert isinstance(model, nn.Module), 'VisionTransformer should inherit from nn.Module'\n"
            "out = model(torch.randn(2, 3, 32, 32))\n"
            "assert out.shape == (2, 10), f'Shape mismatch: {out.shape}'\n"
        },
        {
            "name": "Has cls token and position embedding",
            "code": "\n"
            "import torch\n"
            "model = {fn}(img_size=32, patch_size=8, in_channels=3, num_classes=5, embed_dim=48, depth=1, num_heads=4)\n"
            "assert hasattr(model, 'cls_token'), 'Need self.cls_token'\n"
            "assert hasattr(model, 'pos_embed'), 'Need self.pos_embed'\n"
            "assert hasattr(model, 'num_patches'), 'Need self.num_patches'\n"
            "assert model.num_patches == 16, f'num_patches: {model.num_patches}'\n"
            "assert tuple(model.cls_token.shape) == (1, 1, 48), f'cls_token shape: {tuple(model.cls_token.shape)}'\n"
            "assert tuple(model.pos_embed.shape) == (1, 17, 48), f'pos_embed shape: {tuple(model.pos_embed.shape)}'\n"
        },
        {
            "name": "Works for grayscale images",
            "code": "\n"
            "import torch\n"
            "model = {fn}(img_size=28, patch_size=7, in_channels=1, num_classes=3, embed_dim=32, depth=2, num_heads=4)\n"
            "out = model(torch.randn(4, 1, 28, 28))\n"
            "assert out.shape == (4, 3), f'Shape mismatch: {out.shape}'\n"
        },
        {
            "name": "Identical inputs give identical logits in eval mode",
            "code": "\n"
            "import torch\n"
            "torch.manual_seed(0)\n"
            "model = {fn}(img_size=32, patch_size=8, in_channels=3, num_classes=7, embed_dim=64, depth=2, num_heads=4)\n"
            "model.eval()\n"
            "x = torch.randn(1, 3, 32, 32)\n"
            "batch = x.repeat(2, 1, 1, 1)\n"
            "out = model(batch)\n"
            "assert torch.allclose(out[0], out[1], atol=1e-6), 'Same input should produce same logits in eval mode'\n"
        },
        {
            "name": "Gradient flow to input and special parameters",
            "code": "\n"
            "import torch\n"
            "torch.manual_seed(0)\n"
            "model = {fn}(img_size=32, patch_size=8, in_channels=3, num_classes=10, embed_dim=64, depth=2, num_heads=4)\n"
            "x = torch.randn(2, 3, 32, 32, requires_grad=True)\n"
            "loss = model(x).sum()\n"
            "loss.backward()\n"
            "assert x.grad is not None, 'x.grad is None'\n"
            "assert model.cls_token.grad is not None, 'cls_token should receive gradients'\n"
            "assert model.pos_embed.grad is not None, 'pos_embed should receive gradients'\n"
        },
    ],
}
