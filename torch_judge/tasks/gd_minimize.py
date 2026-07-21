"""Gradient Descent Minimizer task."""

TASK = {
    "title": "Gradient Descent Minimizer",
    "difficulty": "Medium",
    "function_name": "gradient_descent_minimize",
    "hint": (
        "Make a leaf tensor that requires grad: x = x0.clone().detach().requires_grad_(True). "
        "Each step: zero any existing grad, compute y = f(x), call y.backward(), then update "
        "under torch.no_grad(): x -= lr * x.grad. Remember to clear x.grad between steps "
        "(x.grad.zero_()). Return the optimized point detached from the graph."
    ),
    "tests": [
        {
            "name": "1-D quadratic finds the minimum",
            "code": """
import torch
f = lambda x: (x - 3.0) ** 2
x = {fn}(f, torch.tensor([0.0]), lr=0.1, steps=200)
assert torch.allclose(x, torch.tensor([3.0]), atol=1e-2), f'Expected ~3.0, got {x}'
""",
        },
        {
            "name": "Multi-dim bowl converges to center",
            "code": """
import torch
center = torch.tensor([1.0, -2.0, 0.5])
f = lambda x: ((x - center) ** 2).sum()
x = {fn}(f, torch.zeros(3), lr=0.1, steps=300)
assert x.shape == center.shape, f'Shape mismatch: {x.shape}'
assert torch.allclose(x, center, atol=1e-2), f'Expected {center}, got {x}'
""",
        },
        {
            "name": "Lowers the objective value",
            "code": """
import torch
f = lambda x: (x ** 2).sum() + (x * 3).sum()
x0 = torch.tensor([5.0, -4.0])
x = {fn}(f, x0, lr=0.05, steps=300)
assert f(x).item() < f(x0).item(), 'Objective did not decrease'
# closed-form minimum of x^2 + 3x is x = -1.5 per coordinate
assert torch.allclose(x, torch.full((2,), -1.5), atol=1e-2), f'Expected -1.5, got {x}'
""",
        },
        {
            "name": "Returned point is detached with right shape",
            "code": """
import torch
f = lambda x: (x ** 2).sum()
x0 = torch.randn(4)
x = {fn}(f, x0, lr=0.1, steps=50)
assert not x.requires_grad, 'Returned tensor must be detached from the graph'
assert x.shape == x0.shape, f'Shape mismatch: {x.shape} vs {x0.shape}'
""",
        },
        {
            "name": "Does not mutate the input x0",
            "code": """
import torch
f = lambda x: (x ** 2).sum()
x0 = torch.tensor([2.0, -3.0])
x0_copy = x0.clone()
_ = {fn}(f, x0, lr=0.1, steps=100)
assert torch.allclose(x0, x0_copy), 'x0 must not be modified in place'
""",
        },
    ],
}
