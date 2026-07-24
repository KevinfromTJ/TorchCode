"""Optimal Subsequence under GCD Constraint — Alibaba 2026-05-23 (Problem 3).

Non-PyTorch algorithm problem, LeetCode class-mode.
"""

TASK = {
    "title": "Optimal Subsequence (GCD Chain)",
    "difficulty": "Hard",
    "function_name": "Solution",
    "hint": (
        "Minimise the maximum element -> binary search that maximum x. Feasibility is monotone: "
        "if x works, any larger x works. For a fixed x, keep only elements <= x and ask whether a "
        "length-k subsequence exists with adjacent gcd > 1. Two numbers connect iff they share a "
        "prime factor, so factor each a_i into distinct primes and DP: dp_i = 1 + max over p|a_i "
        "of best[p] (longest chain ending with a number containing prime p seen so far); if any "
        "dp_i >= k the value is feasible. Return the smallest feasible value, or -1. (k == 1 -> "
        "just the minimum element; a_i == 1 has no prime factors so it can only stand alone.)"
    ),
    "tests": [
        {
            "name": "Official samples",
            "code": """
sol = {fn}()
assert sol.min_max_element([2, 4, 3, 9, 6], 3) == 6
assert sol.min_max_element([5, 7, 11, 13, 17], 2) == -1
""",
        },
        {
            "name": "k = 1 returns the minimum element",
            "code": """
sol = {fn}()
assert sol.min_max_element([2, 4, 3, 9, 6], 1) == 2
assert sol.min_max_element([9], 1) == 9
""",
        },
        {
            "name": "Chain feasibility edges",
            "code": """
sol = {fn}()
assert sol.min_max_element([6, 10, 15], 2) == 10      # 6-10 share 2
assert sol.min_max_element([2, 3, 4], 3) == -1        # cannot chain 3 with gcd>1 length 3
assert sol.min_max_element([2, 4, 8, 16], 4) == 16    # all even
assert sol.min_max_element([1, 1, 1], 2) == -1        # 1 has no prime factor -> no chain
""",
        },
        {
            "name": "Random vs brute force",
            "code": """
import random
from math import gcd
sol = {fn}()
def brute(a, k):
    n = len(a)
    for limit in sorted(set(a)):
        idx = [i for i in range(n) if a[i] <= limit]
        dp = [1] * len(idx); ok = False
        for x in range(len(idx)):
            for y in range(x):
                if gcd(a[idx[y]], a[idx[x]]) > 1:
                    dp[x] = max(dp[x], dp[y] + 1)
            if dp[x] >= k:
                ok = True
        if ok:
            return limit
    return -1
random.seed(2)
for _ in range(150):
    n = random.randint(2, 8)
    a = [random.randint(1, 30) for _ in range(n)]
    k = random.randint(1, n)
    assert sol.min_max_element(list(a), k) == brute(a, k), (a, k)
""",
        },
        {
            "name": "Larger values factor correctly",
            "code": """
sol = {fn}()
# 1_000_000_007 is prime; pairs must share a factor
assert sol.min_max_element([2, 999999937, 4], 2) == 4        # 999999937 is prime, isolated
assert sol.min_max_element([999983 * 2, 999983 * 3], 2) == 999983 * 3  # share prime 999983
""",
        },
    ],
}
