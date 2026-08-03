"""Dyson Ring Energy Transport — Pinduoduo 2026 Summer 4.12 (Problem 3).

Non-PyTorch algorithm problem, LeetCode class-mode.
A ship starts at `start` on a line with `fuel` units (1 fuel per unit moved).
Passing a depot's position auto-collects its energy. Maximise collected energy.
"""

TASK = {
    "title": "Dyson Ring Energy Transport",
    "difficulty": "Medium",
    "function_name": "Solution",
    "hint": (
        "An optimal route turns direction at most once, so the collected depots form a "
        "contiguous interval [L, R] with L <= start <= R. Reaching both ends costs "
        "(R - L) + min(start - L, R - start) fuel (go to the nearer end first, then sweep "
        "to the far end). Sort depots by position, build a prefix sum of energy, and for "
        "each feasible [L, R] (cost <= fuel) take the interval energy sum; keep the max. "
        "O(M^2) over depot endpoints is plenty; O(M) two-pointer is possible."
    ),
    "tests": [
        {
            "name": "Official sample",
            "code": """
sol = {fn}()
# start=5, fuel=4, depots: pos8/e4, pos4/e7, pos6/e2
# best route 5->4->6 costs (6-4)+min(1,1)=3<=4, collects e7+e2 = 9
assert sol.max_energy(5, 4, [8, 4, 6], [4, 7, 2]) == 9
""",
        },
        {
            "name": "One-directional and no-move cases",
            "code": """
sol = {fn}()
assert sol.max_energy(0, 0, [0, 5], [10, 99]) == 10   # no fuel: only depot at start
assert sol.max_energy(0, 10, [], []) == 0             # no depots
assert sol.max_energy(0, 3, [3], [7]) == 7            # just reach it
assert sol.max_energy(0, 2, [3], [7]) == 0            # cannot reach
assert sol.max_energy(0, 10, [-2, 4], [5, 6]) == 11   # both: span6+min(2,4)=8<=10
""",
        },
        {
            "name": "Turn-around fuel accounting",
            "code": """
sol = {fn}()
# depots at -3 and 3, start 0, energies 5 and 5.
# to get both: span=6, nearer end dist=3 -> cost 6+3=9
assert sol.max_energy(0, 9, [-3, 3], [5, 5]) == 10
assert sol.max_energy(0, 8, [-3, 3], [5, 5]) == 5     # can only reach one side
""",
        },
        {
            "name": "Duplicate positions accumulate",
            "code": """
sol = {fn}()
assert sol.max_energy(0, 0, [0, 0, 0], [1, 2, 3]) == 6
assert sol.max_energy(2, 5, [2, 2, 7], [1, 1, 100]) == 102
""",
        },
        {
            "name": "Random vs brute force",
            "code": """
import random
sol = {fn}()

def brute(start, fuel, positions, energies):
    if not positions:
        return 0
    lo = min([start] + positions); hi = max([start] + positions)
    best = 0
    for L in range(lo, start + 1):
        for R in range(start, hi + 1):
            cost = (R - L) + min(start - L, R - start)
            if cost <= fuel:
                s = sum(e for p, e in zip(positions, energies) if L <= p <= R)
                if s > best:
                    best = s
    return best

random.seed(0)
for _ in range(300):
    m = random.randint(0, 6)
    positions = [random.randint(-10, 10) for _ in range(m)]
    energies = [random.randint(1, 9) for _ in range(m)]
    start = random.randint(-10, 10)
    fuel = random.randint(0, 25)
    assert sol.max_energy(start, fuel, list(positions), list(energies)) == brute(start, fuel, positions, energies), (start, fuel, positions, energies)
""",
        },
    ],
}
