"""Fluctuation Contribution Maximization — Baidu 2026 Autumn 8.6 (Problem 1).

Non-PyTorch algorithm problem, LeetCode class-mode.
Partition the whole sequence into non-empty contiguous segments covering it
entirely. A segment's "fluctuation contribution" is (max - min) * length. Return
the maximum achievable total contribution.
"""

TASK = {
    "title": "Fluctuation Contribution Maximization",
    "difficulty": "Easy",
    "function_name": "Solution",
    "hint": (
        "For any segment, (max - min) <= (global max - global min), so each segment's "
        "contribution <= (M - m) * its length. Summing over a partition gives an upper bound "
        "(M - m) * n, which is achieved by not cutting at all (one whole segment). So the "
        "answer is simply (max(v) - min(v)) * n. Use 64-bit arithmetic."
    ),
    "tests": [
        {
            "name": "Official samples",
            "code": """
sol = {fn}()
assert sol.max_total([2, 5, 1]) == 12
assert sol.max_total([1, 5, 2, 4]) == 16
""",
        },
        {
            "name": "Single element and all equal",
            "code": """
sol = {fn}()
assert sol.max_total([7]) == 0
assert sol.max_total([3, 3, 3, 3]) == 0
""",
        },
        {
            "name": "Random vs interval-DP brute force",
            "code": """
import random
sol = {fn}()

def brute(v):
    n = len(v)
    from functools import lru_cache
    @lru_cache(None)
    def best(i):
        if i == n:
            return 0
        r = 0
        mx = mn = v[i]
        for j in range(i, n):
            mx = max(mx, v[j]); mn = min(mn, v[j])
            r = max(r, (mx - mn) * (j - i + 1) + best(j + 1))
        return r
    return best(0)

random.seed(6)
for _ in range(2000):
    n = random.randint(1, 10)
    v = [random.randint(1, 20) for _ in range(n)]
    assert sol.max_total(list(v)) == brute(tuple(v)), v
""",
        },
        {
            "name": "Large values need 64-bit",
            "code": """
sol = {fn}()
assert sol.max_total([1, 10 ** 9] * 100000) == (10 ** 9 - 1) * 200000
""",
        },
    ],
}
