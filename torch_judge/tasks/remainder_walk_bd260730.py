"""Remainder Walk — Baidu 2026 Autumn 7.30 (Problem 2).

Non-PyTorch algorithm problem, LeetCode class-mode.
A circular sequence a of length n and a modulus M. For a start index s, walk
along the circle accumulating sums; the t-th trajectory point is
S_t = (a[s] + a[s+1] + ... + a[s+t-1]) mod M. L_s is the largest number of steps
t (t <= n) such that S_1..S_t are pairwise distinct. Return the sum of L_s over
all n starts. a is 0-indexed; values may be negative.
"""

TASK = {
    "title": "Remainder Walk",
    "difficulty": "Medium",
    "function_name": "Solution",
    "hint": (
        "Use prefix remainders on the doubled array: P[0]=0, P[i]=(P[i-1]+a[(i-1)%n])%M. "
        "For start s the trajectory points equal P[s+1..s+t] shifted, and distinctness of "
        "S_1..S_t is exactly distinctness of P[s+1..s+t]. So L_s = longest run of distinct "
        "values in P starting at index s+1, capped at n. Slide a window with a 'seen' set and "
        "a monotonic right pointer over j = s+1 (j from 1..n); answer += (r - j). O(n)."
    ),
    "tests": [
        {
            "name": "Official samples",
            "code": """
sol = {fn}()
assert sol.total_length(5, 3, [1, 2, 2, 1, 2]) == 11
assert sol.total_length(4, 5, [5, 0, 5, 0]) == 4
""",
        },
        {
            "name": "Negatives and M = 1",
            "code": """
sol = {fn}()
assert sol.total_length(1, 1, [7]) == 1        # single point S_1 = 0, distinct by itself
assert sol.total_length(2, 1, [3, 4]) == 2     # each start yields exactly one distinct point
""",
        },
        {
            "name": "Random vs brute force",
            "code": """
import random
sol = {fn}()

def brute(n, M, a):
    total = 0; ext = a + a
    for s in range(n):
        seen = set(); cnt = 0; cur = 0
        for t in range(1, n + 1):
            cur = (cur + ext[s + t - 1]) % M
            if cur in seen:
                break
            seen.add(cur); cnt += 1
        total += cnt
    return total

random.seed(4)
for _ in range(4000):
    n = random.randint(1, 12)
    M = random.randint(1, 9)
    a = [random.randint(-9, 9) for _ in range(n)]
    assert sol.total_length(n, M, list(a)) == brute(n, M, a), (n, M, a)
""",
        },
    ],
}
