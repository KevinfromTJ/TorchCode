"""ROC curve + AUC task."""

TASK = {
    "title": "ROC Curve & AUC",
    "difficulty": "Medium",
    "function_name": "roc_auc",
    "hint": (
        "Sort samples by score descending. Sweep the threshold from high to low: at each "
        "point cumulative positives so far are true positives, cumulative negatives are false "
        "positives. tpr = cum_tp / n_pos, fpr = cum_fp / n_neg. Only keep points where the "
        "score changes (collapse ties). Prepend the (0, 0) origin so the curve starts there. "
        "AUC is the area under the curve — integrate tpr over fpr with the trapezoidal rule "
        "(torch.trapz(tpr, fpr)). Return (fpr, tpr, auc)."
    ),
    "tests": [
        {
            "name": "Perfect separation gives AUC = 1",
            "code": """
import torch
scores = torch.tensor([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
labels = torch.tensor([1, 1, 1, 0, 0, 0])
fpr, tpr, auc = {fn}(scores, labels)
assert torch.allclose(auc, torch.tensor(1.0), atol=1e-6), f'AUC should be 1.0, got {auc.item():.4f}'
assert torch.allclose(fpr[0], torch.tensor(0.0)) and torch.allclose(tpr[0], torch.tensor(0.0)), 'Curve must start at (0, 0)'
assert torch.allclose(fpr[-1], torch.tensor(1.0)) and torch.allclose(tpr[-1], torch.tensor(1.0)), 'Curve must end at (1, 1)'
""",
        },
        {
            "name": "Matches rank-based (Mann-Whitney) AUC",
            "code": """
import torch
torch.manual_seed(0)
scores = torch.rand(200)  # distinct scores w.h.p.
labels = (torch.rand(200) > 0.5).long()
_, _, auc = {fn}(scores, labels)
# reference AUC = P(score_pos > score_neg) via ranks
order = torch.argsort(scores)
ranks = torch.empty_like(scores)
ranks[order] = torch.arange(1, scores.numel() + 1, dtype=scores.dtype)
n_pos = (labels == 1).sum().float()
n_neg = (labels == 0).sum().float()
sum_ranks_pos = ranks[labels == 1].sum()
ref = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
assert torch.allclose(auc, ref, atol=1e-4), f'AUC {auc.item():.5f} vs ref {ref.item():.5f}'
""",
        },
        {
            "name": "Invariant to monotonic score transform",
            "code": """
import torch
torch.manual_seed(3)
scores = torch.randn(100)
labels = (torch.rand(100) > 0.4).long()
_, _, auc1 = {fn}(scores, labels)
_, _, auc2 = {fn}(torch.sigmoid(scores * 2.0 + 1.0), labels)  # strictly increasing map
assert torch.allclose(auc1, auc2, atol=1e-5), f'AUC must be invariant to monotonic transforms: {auc1.item():.5f} vs {auc2.item():.5f}'
""",
        },
        {
            "name": "Flipping labels gives 1 - AUC",
            "code": """
import torch
torch.manual_seed(7)
scores = torch.rand(150)
labels = (torch.rand(150) > 0.5).long()
_, _, auc = {fn}(scores, labels)
_, _, auc_flip = {fn}(scores, 1 - labels)
assert torch.allclose(auc + auc_flip, torch.tensor(1.0), atol=1e-4), f'AUC + flipped AUC should be 1, got {(auc + auc_flip).item():.5f}'
""",
        },
        {
            "name": "ROC curve is monotonic and well-formed",
            "code": """
import torch
torch.manual_seed(11)
scores = torch.rand(80)
labels = (torch.rand(80) > 0.5).long()
fpr, tpr, auc = {fn}(scores, labels)
assert fpr.shape == tpr.shape, f'fpr/tpr length mismatch: {fpr.shape} vs {tpr.shape}'
assert (fpr[1:] - fpr[:-1] >= -1e-6).all(), 'FPR must be non-decreasing'
assert (tpr[1:] - tpr[:-1] >= -1e-6).all(), 'TPR must be non-decreasing'
assert 0.0 <= auc.item() <= 1.0, f'AUC out of range: {auc.item()}'
""",
        },
    ],
}
