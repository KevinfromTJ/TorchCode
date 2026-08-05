"""xab — Alibaba 2026-05-09 (Problem 3).

Non-PyTorch algorithm problem, LeetCode class-mode.
Given a fixed x and ranges a in [la, ra], b in [lb, rb] (values up to ~1e9),
choose (a, b) to maximize x XOR a XOR b, then count how many distinct pairs
(a, b) attain that maximum.

Digit DP over the 31 bits (pos 30..0), high bit first. State is a 4-bit mask
of "still tight against a lower/upper bound of a/b" flags; at each bit we
greedily prefer making the XOR bit = 1 (only if any transition can produce
one) and carry the count of pairs consistent with that greedy choice.
"""

TASK = {
    "title": "xab (Max XOR Pair Count)",
    "difficulty": "Hard",
    "function_name": "Solution",
    "hint": (
        "Digit DP from the highest bit. Track 4 tightness flags (a>=la, a<=ra, b>=lb, b<=rb) "
        "as a 16-state mask f[16], starting f[15]=1 (tight on all four bounds). For each bit "
        "pos from 30 down to 0, enumerate the a-bit and b-bit allowed under the current tight "
        "flags, compute the resulting XOR bit x_bit^a_bit^b_bit. If ANY transition yields a "
        "1-bit, greedily keep only those (the max must have this bit set); otherwise keep the "
        "0-bit transitions. Accumulate pair counts into the new mask. Answer = sum(f) at the end."
    ),
    "tests": [
        {
            "name": "Official samples",
            "code": """
sol = {fn}()
assert sol.count_best_pairs(0, 1, 2, 0, 2) == 2
assert sol.count_best_pairs(5, 1, 7, 6, 9) == 2
""",
        },
        {
            "name": "Singleton ranges",
            "code": """
sol = {fn}()
assert sol.count_best_pairs(0, 3, 3, 5, 5) == 1   # only one pair possible
assert sol.count_best_pairs(7, 0, 0, 0, 0) == 1
""",
        },
        {
            "name": "Random vs brute over small ranges",
            "code": """
import random
sol = {fn}()

def brute(x, la, ra, lb, rb):
    best = -1; cnt = 0
    for a in range(la, ra + 1):
        for b in range(lb, rb + 1):
            v = x ^ a ^ b
            if v > best:
                best = v; cnt = 1
            elif v == best:
                cnt += 1
    return cnt

random.seed(11)
for _ in range(3000):
    x = random.randint(0, 63)
    la = random.randint(0, 20); ra = random.randint(la, la + 20)
    lb = random.randint(0, 20); rb = random.randint(lb, lb + 20)
    assert sol.count_best_pairs(x, la, ra, lb, rb) == brute(x, la, ra, lb, rb), (x, la, ra, lb, rb)
""",
        },
        {
            "name": "Large bounds do not crash",
            "code": """
sol = {fn}()
r = sol.count_best_pairs(123456789, 0, 10**9, 0, 10**9)
assert isinstance(r, int) and r >= 1
""",
        },
    ],
}
