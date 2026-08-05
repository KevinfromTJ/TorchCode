"""Diagonal Traversal — Pinduoduo 2026-07-02 (Problem 1).

Non-PyTorch algorithm problem, LeetCode class-mode.
Traverse an m x n matrix by anti-diagonals starting from the bottom-left
corner, alternating the scan direction on successive diagonals.
"""

TASK = {
    "title": "Diagonal Traversal (Bottom-Left)",
    "difficulty": "Easy",
    "function_name": "Solution",
    "hint": (
        "There are m+n-1 diagonals. Index them by g = 0..m+n-2 starting from the bottom-left. "
        "On diagonal g, cells (i, j) satisfy i - j = d where d = (m-1) - g. Row range is "
        "lo = max(0, d) to hi = min(m-1, n-1+d). Walk rows hi->lo when g is even and lo->hi "
        "when g is odd (or vice versa) to alternate direction, taking mat[i][i-d]."
    ),
    "tests": [
        {
            "name": "Official sample 3x3",
            "code": """
sol = {fn}()
assert sol.diagonal_order([[1,2,3],[4,5,6],[7,8,9]]) == [7,4,8,9,5,1,2,6,3]
""",
        },
        {
            "name": "Single row / single column",
            "code": """
sol = {fn}()
assert sol.diagonal_order([[1,2,3,4]]) == [1,2,3,4]
assert sol.diagonal_order([[1],[2],[3]]) == [3,2,1]
assert sol.diagonal_order([[42]]) == [42]
""",
        },
        {
            "name": "Rectangular coverage & length",
            "code": """
import random
sol = {fn}()
random.seed(3)
for _ in range(500):
    m = random.randint(1, 6); n = random.randint(1, 6)
    mat = [[random.randint(0, 99) for _ in range(n)] for _ in range(m)]
    out = sol.diagonal_order(mat)
    assert len(out) == m * n
    assert sorted(out) == sorted(v for row in mat for v in row)
""",
        },
    ],
}
