"""GPU Batch Packing (Fixed Token Buffer) — variant of Problems 2 / 70.

Non-PyTorch algorithm problem, LeetCode class-mode. No discards; a batch is a fixed
token buffer instead of a request count, and consecutive requests inside a batch are
separated by a fixed number of pad tokens (continuous-batching style).
"""

TASK = {
    "title": "GPU Batch Packing (Fixed Token Buffer)",
    "difficulty": "Easy",
    "function_name": "Solution",
    "hint": (
        "Requests arrive in the given order and are packed into fixed-size batches (buffer of B "
        "tokens; unused space is padded). A request costs L tokens, plus G separator pad tokens "
        "for every request AFTER the first in its batch — so a batch of m requests occupies "
        "sum(L) + G*(m-1) <= B. Sweep once, keeping the current batch's used tokens: if the next "
        "request still fits (used + G + L <= B) add it, otherwise open a new batch (used = L). "
        "No reordering, no dropping. Each request fits an empty buffer (L <= B)."
    ),
    "tests": [
        {
            "name": "Basic packing with gaps",
            "code": """
sol = {fn}()
assert sol.min_batches([3, 3, 3], 10, 1) == 2   # 3 | 3+1+3=7 | +1+3=11>10 -> new
assert sol.min_batches([2, 2, 2], 5, 10) == 3   # gap 10 forces one request per batch
""",
        },
        {
            "name": "Gap G changes the answer",
            "code": """
sol = {fn}()
assert sol.min_batches([5, 5], 10, 0) == 1       # 5+5 = 10 fits
assert sol.min_batches([5, 5], 10, 1) == 2       # 5+1+5 = 11 does not
""",
        },
        {
            "name": "Zero-gap sequential packing",
            "code": """
sol = {fn}()
assert sol.min_batches([3, 3, 3], 9, 0) == 1
assert sol.min_batches([3, 3, 3, 3], 9, 0) == 2
assert sol.min_batches([1, 2, 3, 4], 5, 0) == 3   # 1+2 | 3 | 4
""",
        },
        {
            "name": "Requests that individually fill the buffer",
            "code": """
sol = {fn}()
assert sol.min_batches([10, 10, 10], 10, 0) == 3
assert sol.min_batches([7], 7, 5) == 1
assert sol.min_batches([42], 100, 0) == 1
""",
        },
        {
            "name": "Exact fit including the gap",
            "code": """
sol = {fn}()
assert sol.min_batches([4, 4], 9, 1) == 1        # 4+1+4 = 9 == B
assert sol.min_batches([4, 4], 8, 1) == 2        # 4+1+4 = 9 > 8
""",
        },
        {
            "name": "Order matters (no reordering)",
            "code": """
sol = {fn}()
# same multiset, different arrival order -> different batch count
assert sol.min_batches([4, 1, 4, 1], 5, 0) == 2
assert sol.min_batches([1, 1, 4, 4], 5, 0) == 3
""",
        },
        {
            "name": "Property checks over random inputs",
            "code": """
import random
sol = {fn}()
random.seed(1)
for _ in range(300):
    n = random.randint(1, 50)
    b = random.randint(1, 30)
    lengths = [random.randint(1, b) for _ in range(n)]   # each request fits an empty buffer
    g = random.randint(0, 6)
    res = sol.min_batches(list(lengths), b, g)
    assert 1 <= res <= n, 'between one batch and one-per-request'
    # a gap as large as the buffer forces exactly one request per batch
    assert sol.min_batches(list(lengths), b, b) == n
""",
        },
    ],
}
