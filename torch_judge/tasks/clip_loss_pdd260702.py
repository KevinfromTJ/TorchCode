"""CLIP Contrastive Loss — Pinduoduo 2026-07-02 (Problem 4).

PyTorch problem, function-mode.
Given a batch of L2-normalized image embeddings and text embeddings (both
N x D) and a temperature tau, compute the symmetric InfoNCE / CLIP loss:
similarity logits = image @ text.T / tau, then the average of the row-wise
(image->text) and column-wise (text->image) cross-entropy against the
diagonal labels (the i-th image matches the i-th text).
"""

TASK = {
    "title": "CLIP Contrastive Loss",
    "difficulty": "Medium",
    "function_name": "clip_loss",
    "hint": (
        "logits = image_embeds @ text_embeds.T / temperature (shape N x N). The correct match "
        "for row i is column i, so labels = arange(N). Image->text loss is cross_entropy(logits, "
        "labels); text->image loss is cross_entropy(logits.T, labels). Return the average of the "
        "two. F.cross_entropy already does the numerically-stable log-softmax internally."
    ),
    "tests": [
        {
            "name": "Official sample",
            "code": """
import torch
img = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
txt = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
loss = {fn}(img, txt, 1.0)
assert abs(float(loss) - 0.313262) < 1e-5, float(loss)
""",
        },
        {
            "name": "Symmetric in image/text swap",
            "code": """
import torch
torch.manual_seed(0)
img = torch.nn.functional.normalize(torch.randn(8, 16), dim=1)
txt = torch.nn.functional.normalize(torch.randn(8, 16), dim=1)
a = {fn}(img, txt, 0.07)
b = {fn}(txt, img, 0.07)  # logits transpose -> same symmetric loss
assert torch.allclose(a, b, atol=1e-5), (float(a), float(b))
""",
        },
        {
            "name": "Perfect alignment approaches zero loss as tau shrinks",
            "code": """
import torch
n = 6
eye = torch.eye(n)
loss_hot = {fn}(eye, eye, 0.01)
loss_cold = {fn}(eye, eye, 1.0)
assert float(loss_hot) < float(loss_cold)
assert float(loss_hot) < 1e-3
""",
        },
        {
            "name": "Matches manual stable cross-entropy",
            "code": """
import torch
torch.manual_seed(5)
img = torch.nn.functional.normalize(torch.randn(10, 32), dim=1)
txt = torch.nn.functional.normalize(torch.randn(10, 32), dim=1)
tau = 0.05
logits = img @ txt.t() / tau
n = logits.size(0)
def ce(mat):
    m = mat.max(dim=1, keepdim=True).values
    logsum = m.squeeze(1) + torch.log(torch.exp(mat - m).sum(dim=1))
    diag = torch.diagonal(mat)
    return (logsum - diag).mean()
ref = (ce(logits) + ce(logits.t())) / 2
got = {fn}(img, txt, tau)
assert torch.allclose(got, ref, atol=1e-5), (float(got), float(ref))
""",
        },
    ],
}
