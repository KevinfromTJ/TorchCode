"""Message Permutation Reset — Baidu 2026 Autumn 8.6 (Problem 3).

Non-PyTorch algorithm problem, LeetCode class-mode.
Given a lowercase string u of length n and a permutation p (1-indexed) of
1..n, one operation produces w with w[i] = u[p[i]], then sets u = w. Repeating
the operation, return the minimum positive number of operations k that restores
the original string, modulo 1e9+7. Here p is passed 0-indexed (p[i] in 0..n-1).
"""

TASK = {
    "title": "Message Permutation Reset",
    "difficulty": "Medium",
    "function_name": "Solution",
    "hint": (
        "Decompose p into disjoint cycles. Each cycle's characters form a circular string that "
        "rotates one step per operation, so it returns to itself after its minimum rotation "
        "period d (the smallest divisor of the cycle length L such that the cycle string is a "
        "block of length d repeated L/d times; d = 1 if all chars equal). The whole string is "
        "restored at the LCM of all cycle periods. Accumulate the LCM with gcd and take mod "
        "1e9+7 (the true LCM can exceed 64 bits)."
    ),
    "tests": [
        {
            "name": "Official samples",
            "code": """
sol = {fn}()
assert sol.min_reset(2, "xy", [1, 0]) == 2
assert sol.min_reset(3, "zzz", [1, 2, 0]) == 1
assert sol.min_reset(5, "hello", [1, 2, 3, 4, 0]) == 5
""",
        },
        {
            "name": "Identity permutation and single element",
            "code": """
sol = {fn}()
assert sol.min_reset(1, "a", [0]) == 1
assert sol.min_reset(3, "abc", [0, 1, 2]) == 1   # identity -> already unchanged, min positive k = 1
""",
        },
        {
            "name": "Random vs direct-simulation brute force",
            "code": """
import random
sol = {fn}()
MOD = 10 ** 9 + 7

def brute(n, u, p):
    orig = u; cur = u
    for k in range(1, 10 * n + 5):
        cur = ''.join(cur[p[i]] for i in range(n))
        if cur == orig:
            return k % MOD
    return -1

random.seed(6)
for _ in range(3000):
    n = random.randint(1, 8)
    perm = list(range(n)); random.shuffle(perm)
    u = ''.join(random.choice('ab') for _ in range(n))
    assert sol.min_reset(n, u, list(perm)) == brute(n, u, perm), (n, u, perm)
""",
        },
        {
            "name": "Large LCM exceeds 64 bits (mod correctness)",
            "code": """
sol = {fn}()
MOD = 10 ** 9 + 7
# distinct-length cycles with distinct chars -> period == cycle length; LCM of 2,3,5,7,...
import string
lengths = [2, 3, 5, 7, 11, 13, 17, 19, 23]
u = ""; p = []; off = 0
letters = string.ascii_lowercase
for idx, L in enumerate(lengths):
    ch = letters[idx % 26]
    # cycle string with all-distinct rotations: use L distinct chars via (ch, digit)
    for j in range(L):
        u += chr(ord('a') + (j % 26)) if L <= 26 else 'a'
    for j in range(L):
        p.append(off + (j + 1) % L)
    off += L
n = off
expected = 1
import math
for L in lengths:
    expected = expected // math.gcd(expected, L) * L
assert sol.min_reset(n, u, p) == expected % MOD
""",
        },
    ],
}
