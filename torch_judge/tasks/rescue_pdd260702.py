"""Rescue Operation — Pinduoduo 2026-07-02 (Problem 2).

Non-PyTorch algorithm problem, LeetCode class-mode.
There are n guards (1..n). Guard j watches a set of regions; region r is
"protected" while any still-alive guard watches it. A guard standing in an
unwatched region can be killed. Two guards who watch ONLY each other (and are
each other's sole watcher) can be killed together (they cover each other's
blind spot). Peel to a fixed point.

Return "SUCCESS" if every guard can be killed, else the minimum number of
guards that remain.

Input: n and `monitors`, where monitors[j] (0-indexed list, guard j+1) is the
list of 1-indexed regions/guards that guard j+1 watches.
"""

TASK = {
    "title": "Rescue Operation",
    "difficulty": "Medium",
    "function_name": "Solution",
    "hint": (
        "Build monitored_by[r] = set of alive guards watching region r (ignore self-watch). "
        "Repeatedly: (1) kill any guard i with monitored_by[i] empty (no one watches it), "
        "removing its outgoing watches; (2) if none, find a mutual pair i,j where "
        "monitored_by[i]=={j} and monitored_by[j]=={i} and kill both. Iterate to a fixed "
        "point. Return 'SUCCESS' if all dead, else the count of survivors."
    ),
    "tests": [
        {
            "name": "Official sample",
            "code": """
sol = {fn}()
# guard1 watches region2, guard2 watches nothing -> guard2 unwatched, kill it, then guard1 unwatched
assert sol.min_remaining(2, [[2], []]) == "SUCCESS"
""",
        },
        {
            "name": "Mutual pair double-kill",
            "code": """
sol = {fn}()
# 1 and 2 watch only each other -> double kill
assert sol.min_remaining(2, [[2], [1]]) == "SUCCESS"
""",
        },
        {
            "name": "Irreducible cycle",
            "code": """
sol = {fn}()
# 3-cycle: 1->2->3->1, each watched by exactly one other, no mutual pair -> nobody killable
assert sol.min_remaining(3, [[2], [3], [1]]) == 3
""",
        },
        {
            "name": "Random vs recursive brute force",
            "code": """
import random
sol = {fn}()

def brute(n, monitors):
    watch = [[] for _ in range(n + 1)]
    for i in range(n):
        watch[i + 1] = [r for r in monitors[i] if 1 <= r <= n and r != i + 1]
    seen = {}
    full = frozenset(range(1, n + 1))
    def rec(alive):
        if alive in seen:
            return seen[alive]
        mby = {r: set() for r in alive}
        for j in alive:
            for r in watch[j]:
                if r in alive:
                    mby[r].add(j)
        best = len(alive)
        for i in list(alive):
            if not mby[i]:
                best = min(best, rec(alive - {i}))
        for i in list(alive):
            if len(mby[i]) == 1:
                j = next(iter(mby[i]))
                if j in alive and mby[j] == {i}:
                    best = min(best, rec(alive - {i, j}))
        seen[alive] = best
        return best
    return rec(full)

random.seed(13)
for _ in range(1000):
    n = random.randint(1, 6)
    monitors = []
    for i in range(n):
        k = random.randint(0, n)
        monitors.append(random.sample(range(1, n + 1), min(k, n)))
    exp = brute(n, monitors)
    got = sol.min_remaining(n, [list(m) for m in monitors])
    exp_out = "SUCCESS" if exp == 0 else exp
    assert got == exp_out, (n, monitors, got, exp_out)
""",
        },
    ],
}
