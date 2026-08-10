"""Minimum Hedge Value — Baidu 2026 Autumn 8.6 (Problem 2).

Non-PyTorch algorithm problem, LeetCode class-mode.
Split a non-negative integer m into y and m - y (0 <= y <= m). The "hedge value"
is y XOR (m - y). Return the minimum achievable hedge value.
"""

TASK = {
    "title": "Minimum Hedge Value",
    "difficulty": "Medium",
    "function_name": "Solution",
    "hint": (
        "Using a + b = (a XOR b) + 2*(a AND b) with a = y, b = m - y, minimizing a XOR b is "
        "equivalent to maximizing a AND b, which happens at the most balanced split. So the "
        "answer is (m // 2) XOR ((m + 1) // 2). Even m gives 0; m = 2^k - 1 gives m itself. "
        "O(1) per query, and m can be up to 1e18 (needs 64-bit)."
    ),
    "tests": [
        {
            "name": "Official samples",
            "code": """
sol = {fn}()
assert sol.min_hedge(4) == 0
assert sol.min_hedge(5) == 1
assert sol.min_hedge(11) == 3
assert sol.min_hedge(2) == 0
assert sol.min_hedge(7) == 7
""",
        },
        {
            "name": "Zero and one",
            "code": """
sol = {fn}()
assert sol.min_hedge(0) == 0
assert sol.min_hedge(1) == 1
""",
        },
        {
            "name": "Random vs brute force",
            "code": """
import random
sol = {fn}()

def brute(m):
    return min(y ^ (m - y) for y in range(m + 1))

random.seed(7)
for _ in range(2000):
    m = random.randint(0, 400)
    assert sol.min_hedge(m) == brute(m), m
""",
        },
        {
            "name": "Large m up to 1e18",
            "code": """
sol = {fn}()
m = 10 ** 18
assert sol.min_hedge(m) == (m // 2) ^ ((m + 1) // 2)
assert sol.min_hedge(2 ** 60 - 1) == 2 ** 60 - 1   # all-ones -> answer is m itself
""",
        },
    ],
}
