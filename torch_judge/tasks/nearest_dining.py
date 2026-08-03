"""Nearest Free Dining Spot — Alibaba 2026 Summer 4.18 (Problem 2).

Non-PyTorch algorithm problem, LeetCode class-mode.
n forbidden closed integer intervals [l_i, r_i] on the number line; you are at
integer position p. Find the nearest integer x not inside any forbidden
interval and return |x - p|.
"""

TASK = {
    "title": "Nearest Free Dining Spot",
    "difficulty": "Easy",
    "function_name": "Solution",
    "hint": (
        "Intervals are CLOSED and integer, so consecutive forbidden intervals whose gap is 1 "
        "or less merge into one solid forbidden block. Merge overlapping/adjacent intervals; "
        "if p is not inside any block, the answer is 0. Otherwise p sits inside the block "
        "[l, r] and the nearest free integers are l-1 and r+1, so the answer is "
        "min(p-(l-1), (r+1)-p). O(n log n) for the sort."
    ),
    "tests": [
        {
            "name": "Official samples (consistent, closed intervals)",
            "code": """
sol = {fn}()
# p=5, forbidden [1,3],[7,8]: 5 is already free -> 0
assert sol.nearest_free_distance(5, [[1, 3], [7, 8]]) == 0
# p=0, forbidden [-2,1],[2,4],[6,9]: merged [-2,4] blocks 0; free ints -3 or 5 -> 3
assert sol.nearest_free_distance(0, [[-2, 1], [2, 4], [6, 9]]) == 3
""",
        },
        {
            "name": "Inside a single interval — pick nearer edge",
            "code": """
sol = {fn}()
assert sol.nearest_free_distance(10, [[10, 20]]) == 1   # left edge 9 is nearest
assert sol.nearest_free_distance(20, [[10, 20]]) == 1   # right edge 21
assert sol.nearest_free_distance(12, [[10, 20]]) == 3   # 9 at dist 3 vs 21 at dist 9
assert sol.nearest_free_distance(18, [[10, 20]]) == 3   # 21 at dist 3
""",
        },
        {
            "name": "Adjacent intervals merge (gap of 1)",
            "code": """
sol = {fn}()
# [1,3] and [4,6] are contiguous over integers -> solid block [1,6]
assert sol.nearest_free_distance(3, [[1, 3], [4, 6]]) == 3   # nearest free is 0 or 7 -> 3
assert sol.nearest_free_distance(1, [[1, 3], [4, 6]]) == 1   # 0
assert sol.nearest_free_distance(6, [[1, 3], [4, 6]]) == 1   # 7
""",
        },
        {
            "name": "Already free / empty / single point",
            "code": """
sol = {fn}()
assert sol.nearest_free_distance(0, []) == 0
assert sol.nearest_free_distance(-100, [[0, 5]]) == 0    # far away, already free
assert sol.nearest_free_distance(5, [[5, 5]]) == 1       # single-point interval
""",
        },
        {
            "name": "Random vs brute force",
            "code": """
import random
sol = {fn}()

def brute(p, intervals):
    def free(x):
        return all(not (l <= x <= r) for l, r in intervals)
    d = 0
    while True:
        if free(p - d) or free(p + d):
            return d
        d += 1

random.seed(1)
for _ in range(500):
    n = random.randint(0, 6)
    intervals = []
    for _ in range(n):
        a = random.randint(-15, 15); b = a + random.randint(0, 8)
        intervals.append([a, b])
    p = random.randint(-20, 20)
    assert sol.nearest_free_distance(p, [iv[:] for iv in intervals]) == brute(p, intervals), (p, intervals)
""",
        },
    ],
}
