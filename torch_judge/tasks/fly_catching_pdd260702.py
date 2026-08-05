"""Catch the Fly — Pinduoduo 2026-07-02 (Problem 3).

Non-PyTorch algorithm problem, LeetCode class-mode.
n rooms form a functional graph: from room i a fly moves to room a[i]. Placing
a trap in room i costs c[i]. A fly starting anywhere eventually enters a cycle
and loops forever, so it is guaranteed to be caught iff every cycle contains at
least one trapped room. Minimum total cost = sum over cycles of the min c on
that cycle.

Input a is 0-indexed (a[i] in 0..n-1).
"""

TASK = {
    "title": "Catch the Fly",
    "difficulty": "Medium",
    "function_name": "Solution",
    "hint": (
        "In a functional graph every node leads into exactly one cycle. A fly is caught iff "
        "each cycle has a trapped room, and trapping non-cycle rooms is wasteful. Find cycles "
        "by topological peeling: repeatedly remove nodes with in-degree 0; whatever remains "
        "forms the cycles. For each cycle add the minimum c[i] on it. Sum those minima."
    ),
    "tests": [
        {
            "name": "Official sample",
            "code": """
sol = {fn}()
# rooms 1..4, a=[2,4,2,2] (1-indexed) -> 0-indexed a=[1,3,1,1], c=[1,10,2,10]
# only cycle is {1,3} (0-indexed) with costs {10, 10}? -> min 10
assert sol.min_cost([1, 3, 1, 1], [1, 10, 2, 10]) == 10
""",
        },
        {
            "name": "Self-loops and single node",
            "code": """
sol = {fn}()
assert sol.min_cost([0], [5]) == 5                 # single self-loop
assert sol.min_cost([0, 1], [3, 7]) == 10          # two separate self-loops
assert sol.min_cost([1, 0], [4, 9]) == 4           # one 2-cycle -> min(4,9)
""",
        },
        {
            "name": "Random vs cycle-detection brute",
            "code": """
import random
sol = {fn}()

def brute(a, c):
    n = len(a)
    incycle = [False] * n
    for i in range(n):
        x = a[i]; steps = 0; onc = False
        while steps <= n:
            if x == i:
                onc = True; break
            x = a[x]; steps += 1
        incycle[i] = onc
    visited = [False] * n; ans = 0
    for i in range(n):
        if incycle[i] and not visited[i]:
            mn = c[i]; visited[i] = True; x = a[i]
            while x != i:
                visited[x] = True; mn = min(mn, c[x]); x = a[x]
            ans += mn
    return ans

random.seed(17)
for _ in range(2000):
    n = random.randint(1, 10)
    a = [random.randint(0, n - 1) for _ in range(n)]
    c = [random.randint(1, 50) for _ in range(n)]
    assert sol.min_cost(a, c) == brute(a, c), (a, c)
""",
        },
    ],
}
