"""Thorn Forest Optimal Cutting — Alibaba 2026-05-23 (Problem 1).

Non-PyTorch algorithm problem, LeetCode class-mode.
"""

TASK = {
    "title": "Thorn Forest Optimal Cutting",
    "difficulty": "Easy",
    "function_name": "Solution",
    "hint": (
        "A failed attempt only wastes sharpness, so an optimal plan never attempts a thorn it "
        "cannot cut. If the j-th successful cut (1-indexed) happens at sharpness K-(j-1), harder "
        "thorns should be cut earlier. Sort hardness descending and sweep: keeping cnt successes "
        "so far, the current sharpness is K-cnt; if K-cnt >= a_i, cut it (cnt += 1), otherwise "
        "skip (every later thorn is softer but sharpness only drops). Return cnt."
    ),
    "tests": [
        {
            "name": "Official samples",
            "code": """
sol = {fn}()
assert sol.max_cuts([2, 1, 4, 10, 3], 5) == 4
assert sol.max_cuts([10, 10], 1) == 0
assert sol.max_cuts([3, 3, 3], 3) == 1
""",
        },
        {
            "name": "Greedy edge cases",
            "code": """
sol = {fn}()
assert sol.max_cuts([5, 4, 3, 2, 1], 3) == 3   # cut 3, then 2, then 1
assert sol.max_cuts([1, 1, 1, 1], 2) == 2      # sharpness 2 -> two cuts of hardness 1
assert sol.max_cuts([100], 1) == 0             # too hard for the only attempt
assert sol.max_cuts([1], 1) == 1
""",
        },
        {
            "name": "Order independence (unsorted input)",
            "code": """
sol = {fn}()
assert sol.max_cuts([3, 10, 2, 4, 1], 5) == 4
assert sol.max_cuts([1, 2, 4, 10, 3], 5) == 4
""",
        },
        {
            "name": "Random vs brute force",
            "code": """
import itertools, random
sol = {fn}()
def brute(a, k):
    best = 0
    for r in range(len(a) + 1):
        for comb in itertools.permutations(a, r):
            if all(k - j >= x for j, x in enumerate(comb)):
                best = max(best, r)
    return best
random.seed(0)
for _ in range(60):
    n = random.randint(1, 6)
    a = [random.randint(1, 8) for _ in range(n)]
    k = random.randint(1, 8)
    assert sol.max_cuts(list(a), k) == brute(a, k), (a, k)
""",
        },
    ],
}
