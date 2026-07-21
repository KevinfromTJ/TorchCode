"""Side-Scroller Coin Catching — Pinduoduo 2026-07-19 (Problem 3).

Non-PyTorch algorithm problem, LeetCode class-mode.
"""

TASK = {
    "title": "Side-Scroller Coin Catching",
    "difficulty": "Medium",
    "function_name": "Solution",
    "hint": (
        "Catching A then B requires T_B - T_A > |X_B - X_A|, which splits into "
        "T_B - X_B > T_A - X_A and T_B + X_B > T_A + X_A. Map each coin to (u, v) = "
        "(T - X, T + X); the answer is the largest subset that is strictly increasing in BOTH "
        "u and v. Sort by u ascending, and by v DESCENDING when u ties (so equal-u coins can't "
        "chain), then take the strictly-increasing longest-increasing-subsequence over v "
        "(bisect_left keeps it strict). The starting position is free, so any coin can be first."
    ),
    "tests": [
        {
            "name": "Official sample",
            "code": """
sol = {fn}()
assert sol.max_coins([[1, 2], [5, 5], [2, 3], [7, 4], [6, 8]]) == 3
""",
        },
        {
            "name": "Single coin",
            "code": """
sol = {fn}()
assert sol.max_coins([[5, 5]]) == 1
assert sol.max_coins([[1000000000, -1000000000]]) == 1
""",
        },
        {
            "name": "Duplicate coins at the same time and place",
            "code": """
sol = {fn}()
# simultaneous coins at the same coordinate: at most one can be taken
assert sol.max_coins([[5, 5], [5, 5]]) == 1
assert sol.max_coins([[5, 5], [5, 5], [5, 5]]) == 1
""",
        },
        {
            "name": "Same instant, different coordinates",
            "code": """
sol = {fn}()
# cannot be in two places at one time -> only one
assert sol.max_coins([[3, 1], [3, 9]]) == 1
""",
        },
        {
            "name": "Fully chainable sequence",
            "code": """
sol = {fn}()
# same x, times spaced by 1 (> distance 0) -> all catchable
assert sol.max_coins([[1, 0], [2, 0], [3, 0], [4, 0]]) == 4
assert sol.max_coins([[1, 0], [10, 0], [100, 0]]) == 3
""",
        },
        {
            "name": "Boundary: strictly-greater required",
            "code": """
sol = {fn}()
# each gap in time exactly equals the distance -> cannot chain, answer 1
assert sol.max_coins([[1, 5], [2, 4], [3, 3]]) == 1
""",
        },
        {
            "name": "All coins on one u-diagonal",
            "code": """
sol = {fn}()
# all share T - X, so u never strictly increases -> at most one
assert sol.max_coins([[10, 10], [1, 1], [5, 5], [2, 2]]) == 1
""",
        },
        {
            "name": "Unsorted input handled",
            "code": """
sol = {fn}()
assert sol.max_coins([[7, 4], [1, 2], [6, 8], [2, 3], [5, 5]]) == 3
""",
        },
    ],
}
