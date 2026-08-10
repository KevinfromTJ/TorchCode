"""Max Product Operation — Baidu 2026 Autumn 7.30 (Problem 3).

Non-PyTorch algorithm problem, LeetCode class-mode.
A sequence of positive integers. You must perform exactly k operations; each
operation decrements one element that is currently > 1 (if every element equals
1 you stop early). Maximize the product of the sequence, returned modulo 1e9+7.
"""

TASK = {
    "title": "Max Product Operation",
    "difficulty": "Hard",
    "function_name": "Solution",
    "hint": (
        "Greedy: always decrement the current maximum, because (x-1)/x >= (y-1)/y for x >= y, "
        "so a decrement hurts the largest element least. Sort ascending; total = sum(x-1). If "
        "k >= total, the answer is 1. Otherwise, from the top, lower the largest `cnt` elements "
        "together level by level while affordable (cost to drop cnt elements to the next value = "
        "(level - next) * cnt). With remaining k, q, r = divmod(k, cnt); high = level - q; the "
        "product is prefix(untouched) * high^(cnt-r) * (high-1)^r, all mod 1e9+7."
    ),
    "tests": [
        {
            "name": "Official samples",
            "code": """
sol = {fn}()
assert sol.max_product([2, 2, 3], 3) == 2
assert sol.max_product([5, 1, 3, 2], 2) == 18
assert sol.max_product([10], 100) == 1
""",
        },
        {
            "name": "k = 0 keeps the product; all ones",
            "code": """
sol = {fn}()
MOD = 10 ** 9 + 7
assert sol.max_product([3, 4], 0) == 12
assert sol.max_product([1, 1, 1], 5) == 1
""",
        },
        {
            "name": "Random vs exhaustive brute force",
            "code": """
import random
sol = {fn}()
MOD = 10 ** 9 + 7

def brute(a, k):
    best = [-1]
    seen = {}
    def dfs(state, kk):
        state = tuple(sorted(state))
        if all(x == 1 for x in state) or kk == 0:
            p = 1
            for x in state:
                p *= x
            best[0] = max(best[0], p)
            return
        if (state, kk) in seen:
            return
        seen[(state, kk)] = True
        for i in range(len(state)):
            if state[i] > 1:
                ns = list(state); ns[i] -= 1
                dfs(ns, kk - 1)
    dfs(a, k)
    return best[0] % MOD

random.seed(5)
for _ in range(3000):
    n = random.randint(1, 5)
    a = [random.randint(1, 6) for _ in range(n)]
    k = random.randint(0, 9)
    assert sol.max_product(list(a), k) == brute(a, k), (a, k)
""",
        },
        {
            "name": "Large k stays under the modulus",
            "code": """
sol = {fn}()
MOD = 10 ** 9 + 7
r = sol.max_product([10 ** 9, 10 ** 9, 10 ** 9], 0)
assert r == (10 ** 9 % MOD) ** 3 % MOD
assert 0 <= sol.max_product([10 ** 9] * 5, 10 ** 18) < MOD
""",
        },
    ],
}
