"""Grid Walk Flip — Baidu 2026 Autumn 7.23 (Problem 3).

Non-PyTorch algorithm problem, LeetCode class-mode.
An infinite 1D grid is indexed from 1. Initially prime-numbered grids are black
(1) and the rest white (0). There are n people; person i (1-indexed) performs a
walk consuming the first i characters of the binary string s. Each walk starts
at grid 1; for character '0' the walker jumps to the next WHITE grid strictly to
the right of the current one, for '1' to the next BLACK grid. After finishing
their walk, the person FLIPS the color of the grid they end on. Color changes
persist and affect later walkers. Return (count, sorted_list) of grids whose
final color differs from the initial color.
"""

TASK = {
    "title": "Grid Walk Flip",
    "difficulty": "Hard",
    "function_name": "Solution",
    "hint": (
        "Track only flip parity per grid on top of the prime base color: color(g) = "
        "is_prime(g) XOR parity[g]. Simulate each of the n people: start cur=1, and for each "
        "char scan rightward (g = cur+1, cur+2, ...) to the first grid whose current color "
        "matches the target (white for '0', black for '1'), then move there. After the walk, "
        "flip parity of the final grid. Report grids with odd parity, sorted."
    ),
    "tests": [
        {
            "name": "Official samples",
            "code": """
sol = {fn}()
assert sol.changed_grids(1, "0") == (1, [4])
assert sol.changed_grids(2, "11") == (2, [2, 5])
""",
        },
        {
            "name": "Return type and single person",
            "code": """
sol = {fn}()
cnt, lst = sol.changed_grids(1, "1")
assert cnt == len(lst)
assert lst == sorted(lst)
assert cnt == 1
""",
        },
        {
            "name": "Random vs independent sieve brute force",
            "code": """
import random
sol = {fn}()

def sieve_brute(n, s, LIMIT=20000):
    isp = [True] * (LIMIT + 1)
    isp[0] = isp[1] = False
    for i in range(2, int(LIMIT ** 0.5) + 1):
        if isp[i]:
            for j in range(i * i, LIMIT + 1, i):
                isp[j] = False
    color = {}
    def col(g):
        if g in color:
            return color[g]
        return 1 if isp[g] else 0
    for i in range(1, n + 1):
        cur = 1
        for c in s[:i]:
            need = 1 if c == '1' else 0
            g = cur + 1
            while col(g) != need:
                g += 1
            cur = g
        color[cur] = 1 - col(cur)
    changed = sorted(g for g in color if color[g] != (1 if isp[g] else 0))
    return len(changed), changed

random.seed(9)
for _ in range(400):
    n = random.randint(1, 25)
    s = ''.join(random.choice('01') for _ in range(n))
    assert sol.changed_grids(n, s) == sieve_brute(n, s), (n, s)
""",
        },
    ],
}
