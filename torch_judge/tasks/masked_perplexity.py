"""Masked Perplexity task."""

TASK = {
    "title": "Masked Perplexity",
    "difficulty": "Medium",
    "function_name": "masked_perplexity",
    "hint": (
        "Perplexity = exp(mean token NLL). Compute log-probs stably with the logsumexp "
        "trick: log_probs = logits - logsumexp(logits, dim=-1, keepdim=True). Pick the "
        "target log-prob per position WITHOUT torch.gather — flatten to (B*T, V) and index "
        "with log_probs[arange(B*T), targets.reshape(-1)]. Average -log_prob over the mask "
        "only: nll = -(tgt_logp * mask).sum() / mask.sum(), then return nll.exp()."
    ),
    "tests": [
        {
            "name": "Matches reference perplexity",
            "code": """
import torch
torch.manual_seed(0)
B, T, V = 3, 6, 12
logits = torch.randn(B, T, V)
targets = torch.randint(0, V, (B, T))
mask = torch.tensor([
    [1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 0],
], dtype=torch.float32)
out = {fn}(logits, targets, mask)
# reference: token-level NLL averaged over masked positions, then exp
ce = torch.nn.functional.cross_entropy(
    logits.reshape(B * T, V), targets.reshape(B * T), reduction='none'
).reshape(B, T)
ref = torch.exp((ce * mask).sum() / mask.sum())
assert out.dim() == 0, 'Perplexity must be a scalar'
assert torch.allclose(out, ref, atol=1e-4), f'Mismatch: {out.item():.5f} vs {ref.item():.5f}'
""",
        },
        {
            "name": "Masked positions are ignored",
            "code": """
import torch
torch.manual_seed(1)
B, T, V = 2, 5, 8
logits = torch.randn(B, T, V)
targets = torch.randint(0, V, (B, T))
mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.float32)
out1 = {fn}(logits, targets, mask)
# corrupt logits/targets only at masked positions — result must not change
logits2 = logits.clone()
logits2[mask == 0] = 100.0
targets2 = targets.clone()
targets2[mask == 0] = 0
out2 = {fn}(logits2, targets2, mask)
assert torch.allclose(out1, out2, atol=1e-5), 'Masked tokens must not affect perplexity'
""",
        },
        {
            "name": "Full mask equals exp(mean CE)",
            "code": """
import torch
torch.manual_seed(2)
B, T, V = 4, 7, 10
logits = torch.randn(B, T, V)
targets = torch.randint(0, V, (B, T))
mask = torch.ones(B, T)
out = {fn}(logits, targets, mask)
ref = torch.exp(torch.nn.functional.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T)))
assert torch.allclose(out, ref, atol=1e-4), f'{out.item():.5f} vs {ref.item():.5f}'
""",
        },
        {
            "name": "Numerical stability with large logits",
            "code": """
import torch
logits = torch.tensor([[[1000., 0., 0.], [0., 1000., 0.]]])
targets = torch.tensor([[0, 1]])
mask = torch.ones(1, 2)
out = {fn}(logits, targets, mask)
assert not torch.isnan(out) and not torch.isinf(out), 'PPL blew up on large logits'
assert out.item() < 1.01, f'Confident correct predictions should give PPL ~ 1, got {out.item():.4f}'
""",
        },
        {
            "name": "No torch.gather used",
            "code": """
import inspect, torch
src = inspect.getsource({fn})
assert 'gather' not in src, 'Implement the target lookup without torch.gather'
""",
        },
        {
            "name": "Gradient flows to logits",
            "code": """
import torch
logits = torch.randn(2, 4, 6, requires_grad=True)
targets = torch.randint(0, 6, (2, 4))
mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)
out = {fn}(logits, targets, mask)
out.backward()
assert logits.grad is not None, 'Gradient did not flow to logits'
""",
        },
    ],
}
