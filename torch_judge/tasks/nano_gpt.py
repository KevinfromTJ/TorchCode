"""Minimal GPT (nanoGPT-style) full model: structure + training + inference."""

TASK = {
    "title": "Minimal GPT (nanoGPT): Model + Train + Generate",
    "difficulty": "Hard",
    "function_name": "GPT",
    "hint": (
        "nanoGPT-style decoder-only Transformer. __init__(vocab_size, block_size, "
        "n_layer, n_head, n_embd). Token embedding + learnable positional embedding, "
        "n_layer pre-norm blocks (causal self-attention + MLP with 4x GELU), final "
        "LayerNorm, then lm_head Linear(n_embd, vocab_size). "
        "forward(idx, targets=None) -> (logits, loss); logits shape (B, T, vocab_size); "
        "loss is F.cross_entropy over the flattened logits/targets, or None if targets is None. "
        "Attention must be causal. generate(idx, max_new_tokens) autoregressively appends "
        "argmax tokens (greedy), cropping context to the last block_size tokens each step."
    ),
    "tests": [
        {
            "name": "Forward logits shape and loss=None without targets",
            "code": """
import torch, torch.nn as nn
torch.manual_seed(0)
model = {fn}(vocab_size=17, block_size=8, n_layer=2, n_head=2, n_embd=32)
assert isinstance(model, nn.Module), 'GPT should inherit from nn.Module'
idx = torch.randint(0, 17, (3, 5))
logits, loss = model(idx)
assert logits.shape == (3, 5, 17), f'Logits shape: {tuple(logits.shape)}'
assert loss is None, 'loss must be None when targets is not given'
""",
        },
        {
            "name": "Training loss is a scalar and gradients flow to all params",
            "code": """
import torch
torch.manual_seed(0)
model = {fn}(vocab_size=17, block_size=8, n_layer=2, n_head=2, n_embd=32)
idx = torch.randint(0, 17, (4, 8))
targets = torch.randint(0, 17, (4, 8))
logits, loss = model(idx, targets)
assert logits.shape == (4, 8, 17), f'Logits shape: {tuple(logits.shape)}'
assert loss.dim() == 0, 'loss must be a scalar'
assert loss.item() > 0, 'cross-entropy loss should be positive'
# random init: loss near ln(vocab)
import math
assert abs(loss.item() - math.log(17)) < 1.5, f'loss {loss.item():.3f} far from ln(17)'
loss.backward()
n_total = sum(1 for p in model.parameters())
n_grad = sum(1 for p in model.parameters() if p.grad is not None)
assert n_grad == n_total, f'Only {n_grad}/{n_total} params got gradients'
""",
        },
        {
            "name": "Loss actually decreases over a few training steps (overfit tiny batch)",
            "code": """
import torch
torch.manual_seed(0)
model = {fn}(vocab_size=13, block_size=16, n_layer=2, n_head=2, n_embd=32)
idx = torch.randint(0, 13, (2, 16))
targets = torch.randint(0, 13, (2, 16))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
_, first = model(idx, targets)
for _ in range(50):
    opt.zero_grad()
    _, loss = model(idx, targets)
    loss.backward()
    opt.step()
assert loss.item() < first.item() - 0.5, f'loss did not decrease: {first.item():.3f} -> {loss.item():.3f}'
""",
        },
        {
            "name": "Causal masking — a future token cannot change earlier logits",
            "code": """
import torch
torch.manual_seed(0)
model = {fn}(vocab_size=17, block_size=8, n_layer=2, n_head=2, n_embd=32)
model.eval()
idx = torch.randint(0, 17, (1, 8))
with torch.no_grad():
    out1, _ = model(idx)
    idx2 = idx.clone()
    idx2[0, 5:] = (idx2[0, 5:] + 3) % 17   # perturb positions 5..7
    out2, _ = model(idx2)
assert torch.allclose(out1[:, :5], out2[:, :5], atol=1e-5), 'Future tokens changed earlier logits — not causal'
""",
        },
        {
            "name": "generate() extends sequence deterministically (greedy)",
            "code": """
import torch
torch.manual_seed(0)
model = {fn}(vocab_size=17, block_size=8, n_layer=2, n_head=2, n_embd=32)
model.eval()
idx = torch.randint(0, 17, (2, 3))
out = model.generate(idx, max_new_tokens=5)
assert out.shape == (2, 8), f'generate shape: {tuple(out.shape)}'
assert torch.equal(out[:, :3], idx), 'generate must keep the original prompt as a prefix'
assert out.min().item() >= 0 and out.max().item() < 17, 'generated ids out of vocab range'
# greedy generation is deterministic
out2 = model.generate(idx, max_new_tokens=5)
assert torch.equal(out, out2), 'greedy generate should be deterministic'
""",
        },
        {
            "name": "generate() crops context beyond block_size",
            "code": """
import torch
torch.manual_seed(0)
model = {fn}(vocab_size=17, block_size=8, n_layer=2, n_head=2, n_embd=32)
model.eval()
idx = torch.randint(0, 17, (1, 8))   # already at block_size
out = model.generate(idx, max_new_tokens=6)  # must not crash on > block_size context
assert out.shape == (1, 14), f'generate shape: {tuple(out.shape)}'
assert torch.equal(out[:, :8], idx), 'prompt prefix must be preserved'
""",
        },
    ],
}
