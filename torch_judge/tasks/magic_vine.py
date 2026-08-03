"""Magic Vine Cutting — Alibaba 2026 Summer 7.23 (Problem 2).

Non-PyTorch algorithm problem, LeetCode class-mode.
A vine is a_1..a_n. The remaining vine is always a prefix. One operation:
find the LAST occurrence of the current prefix's max (or min) and cut that
position together with everything to its right (new prefix length = pos-1).
Return the minimum number of operations until segment 1 is cut (length 0).
"""

TASK = {
    "title": "Magic Vine Cutting",
    "difficulty": "Medium",
    "function_name": "Solution",
    "hint": (
        "The remaining vine is always a prefix, so let dp[k] = min ops to fully cut a prefix "
        "of length k (dp[0] = 0). For prefix length k, one op cuts at the last position of the "
        "max -> new length maxpos[k]-1, or the last position of the min -> minpos[k]-1. So "
        "dp[k] = 1 + min(dp[maxpos[k]-1], dp[minpos[k]-1]). Precompute maxpos[k]/minpos[k] "
        "(1-indexed last occurrence of the running max/min over a[0..k-1]) in one left-to-right "
        "pass. Answer is dp[n]. O(n)."
    ),
    "tests": [
        {
            "name": "Official samples",
            "code": """
sol = {fn}()
assert sol.min_operations([1, 3, 2]) == 1      # min is 1 at pos1 -> cut everything at once
assert sol.min_operations([4, 2, 4, 1]) == 2
""",
        },
        {
            "name": "Single / two elements",
            "code": """
sol = {fn}()
assert sol.min_operations([5]) == 1            # only one op possible
assert sol.min_operations([1, 2]) == 1         # min 1 at pos1 -> cut both
assert sol.min_operations([2, 1]) == 1         # max 2 at pos1 -> cut both
""",
        },
        {
            "name": "Max/min at first position cuts everything in one op",
            "code": """
sol = {fn}()
assert sol.min_operations([9, 1, 5, 3]) == 1   # max 9 at pos1
assert sol.min_operations([1, 9, 5, 3]) == 1   # min 1 at pos1
assert sol.min_operations([5, 5, 5]) == 3      # max==min, last-occ is always pos k -> peel one at a time
""",
        },
        {
            "name": "Ascending and descending",
            "code": """
sol = {fn}()
# ascending [1,2,3,4,5]: min is at pos1 always once we can reach prefix len k, min last-occ=1
assert sol.min_operations([1, 2, 3, 4, 5]) == 1
# descending [5,4,3,2,1]: max last-occ=1 -> one op
assert sol.min_operations([5, 4, 3, 2, 1]) == 1
""",
        },
        {
            "name": "Random vs brute force (recursive)",
            "code": """
import random
sol = {fn}()

def brute(a):
    memo = {}
    def solve(k):
        if k == 0:
            return 0
        if k in memo:
            return memo[k]
        pref = a[:k]
        mx = max(pref); mn = min(pref)
        mpos = max(i for i in range(k) if pref[i] == mx) + 1
        npos = max(i for i in range(k) if pref[i] == mn) + 1
        memo[k] = 1 + min(solve(mpos - 1), solve(npos - 1))
        return memo[k]
    return solve(len(a))

random.seed(2)
for _ in range(500):
    n = random.randint(1, 12)
    a = [random.randint(1, 5) for _ in range(n)]
    assert sol.min_operations(list(a)) == brute(a), a
""",
        },
    ],
}
