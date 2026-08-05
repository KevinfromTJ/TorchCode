"""Signed Permutation — Alibaba 2026-05-09 (Problem 1).

Non-PyTorch algorithm problem, LeetCode class-mode.
Given n, construct a length-n array where |a_i| is a permutation of 1..n,
every a_i != 0, and the total sum is 0. Return [] if impossible.
Impossible exactly when n % 4 in {1, 2}.
"""

TASK = {
    "title": "Signed Permutation (Sum Zero)",
    "difficulty": "Medium",
    "function_name": "Solution",
    "hint": (
        "Sum of 1..n is n(n+1)/2. Assigning +/- signs, the achievable sums have the same "
        "parity as that total, so a zero sum is only possible when n(n+1)/2 is even, i.e. "
        "n % 4 in {0, 3}. Construction: consecutive quadruples (x, -(x+1), -(x+2), (x+3)) "
        "each sum to 0. For n % 4 == 0 tile from x=1. For n % 4 == 3 emit [1, 2, -3] first "
        "(sum 0) then tile quadruples from x=4. Return [] when n % 4 in {1, 2}."
    ),
    "tests": [
        {
            "name": "Official samples",
            "code": """
sol = {fn}()
assert sol.construct(2) == []
r3 = sol.construct(3)
assert sorted(abs(v) for v in r3) == [1, 2, 3] and sum(r3) == 0 and all(v != 0 for v in r3)
r4 = sol.construct(4)
assert sorted(abs(v) for v in r4) == [1, 2, 3, 4] and sum(r4) == 0 and all(v != 0 for v in r4)
""",
        },
        {
            "name": "Impossible cases",
            "code": """
sol = {fn}()
for n in [1, 2, 5, 6, 9, 10, 13, 14]:
    assert sol.construct(n) == [], n
""",
        },
        {
            "name": "Valid construction for feasible n",
            "code": """
sol = {fn}()
for n in [3, 4, 7, 8, 11, 12, 100, 103]:
    res = sol.construct(n)
    assert len(res) == n, n
    assert sorted(abs(v) for v in res) == list(range(1, n + 1)), n
    assert sum(res) == 0, n
    assert all(v != 0 for v in res), n
""",
        },
        {
            "name": "Large n stress",
            "code": """
sol = {fn}()
for n in [10000, 10003, 99999, 100000]:
    res = sol.construct(n)
    if n % 4 in (1, 2):
        assert res == [], n
    else:
        assert len(res) == n and sum(res) == 0
        assert sorted(abs(v) for v in res) == list(range(1, n + 1))
""",
        },
    ],
}
