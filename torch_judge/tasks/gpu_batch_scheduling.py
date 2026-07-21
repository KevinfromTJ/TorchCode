"""GPU Batch Scheduling — Minimum Batches — Pinduoduo 2026-07-19 (Problem 2).

Non-PyTorch algorithm problem, LeetCode class-mode.
"""

TASK = {
    "title": "GPU Batch Scheduling (Min Batches)",
    "difficulty": "Hard",
    "function_name": "Solution",
    "hint": (
        "Sort lengths ascending; a batch is a contiguous window with max-min <= K and size <= C. "
        "You may drop up to M requests. DP over sorted prefix: f[i][j] = min batches using the first "
        "i requests having dropped exactly j. Two-pointer gives left_i (smallest index with "
        "a[i]-a[left]<=K); let s = max(left-1, i-C). Transitions: drop request i -> f[i-1][j-1]; "
        "keep i as the last request of a new batch -> f[max(s,j)][j]+1 (no extra drops), or "
        "g[s-1][j-1]+1 where g is the prefix-min along the diagonal f[p][q] with p-q constant. "
        "Answer = min over 0<=j<=M of f[N][j]."
    ),
    "tests": [
        {
            "name": "Official samples",
            "code": """
sol = {fn}()
assert sol.min_batches([10, 11, 15, 16, 17], 1, 1, 2) == 2
assert sol.min_batches([1, 100, 2, 200], 2, 10, 3) == 1
""",
        },
        {
            "name": "Order independence (unsorted input)",
            "code": """
sol = {fn}()
assert sol.min_batches([17, 10, 16, 11, 15], 1, 1, 2) == 2
assert sol.min_batches([200, 1, 100, 2], 2, 10, 3) == 1
""",
        },
        {
            "name": "No discards allowed (M=0)",
            "code": """
sol = {fn}()
assert sol.min_batches([1, 2, 3, 4, 5, 6], 0, 1, 2) == 3
assert sol.min_batches([10, 11, 15, 16, 17], 0, 1, 2) == 3
""",
        },
        {
            "name": "Discarding reduces batch count",
            "code": """
sol = {fn}()
# same list as above but 2 discards let us drop the awkward middle points
assert sol.min_batches([1, 2, 3, 4, 5, 6], 2, 1, 2) == 2
""",
        },
        {
            "name": "K=0 forces equal-length batches",
            "code": """
sol = {fn}()
assert sol.min_batches([1, 2, 3, 4], 0, 0, 4) == 4      # all distinct -> 4 batches
assert sol.min_batches([3, 3, 3, 3], 0, 0, 2) == 2      # capacity splits equal lengths
assert sol.min_batches([5, 5, 5], 0, 0, 3) == 1
""",
        },
        {
            "name": "Capacity C dominates when K is large",
            "code": """
sol = {fn}()
assert sol.min_batches([1, 2, 3, 4, 5, 6], 0, 100, 3) == 2   # ceil(6/3)
assert sol.min_batches([1, 1, 1, 1, 1], 3, 0, 10) == 1
""",
        },
        {
            "name": "Discard everything (M >= N)",
            "code": """
sol = {fn}()
assert sol.min_batches([1, 2, 3, 4, 5], 5, 0, 1) == 0
assert sol.min_batches([7], 1, 0, 1) == 0
""",
        },
        {
            "name": "Single request",
            "code": """
sol = {fn}()
assert sol.min_batches([5], 0, 0, 1) == 1
assert sol.min_batches([42], 0, 1000000000, 5) == 1
""",
        },
    ],
}
