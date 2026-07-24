"""Multi-Constraint Element Matching Count — Alibaba 2026-05-23 (Problem 2).

Non-PyTorch algorithm problem, LeetCode class-mode.
"""

TASK = {
    "title": "Multi-Constraint Element Matching",
    "difficulty": "Medium",
    "function_name": "Solution",
    "hint": (
        "Count ordered pairs (i, j) with 1<=i<=j<=n and a_i == b[c_j]. Fix the right endpoint j "
        "and sweep left to right, maintaining a hash map of how many times each value has "
        "appeared in the prefix a[1..j]. At position j: first insert a[j] into the map, then let "
        "x = b[c_j] (c_j is 1-indexed, so b[c_j - 1] in 0-indexed code) and add map[x] to the "
        "answer. O(n) per test with an average-O(1) hash map."
    ),
    "tests": [
        {
            "name": "Official samples",
            "code": """
sol = {fn}()
assert sol.count_pairs([1, 2, 1, 2, 1], [2, 1, 3, 1, 2], [2, 1, 5, 4, 3]) == 5
assert sol.count_pairs([7, 7, 7], [1, 2, 7], [3, 3, 3]) == 6
""",
        },
        {
            "name": "Small and edge cases",
            "code": """
sol = {fn}()
assert sol.count_pairs([1], [1], [1]) == 1
assert sol.count_pairs([5, 5], [5, 9], [1, 1]) == 3          # (0,0),(0,1),(1,1)
assert sol.count_pairs([1, 2, 3], [4, 5, 6], [1, 2, 3]) == 0  # no matches
""",
        },
        {
            "name": "1-indexed c is handled",
            "code": """
sol = {fn}()
# c_j points into b with 1-based indices; must read b[c_j - 1]
assert sol.count_pairs([9, 9], [9, 0], [1, 1]) == 3
assert sol.count_pairs([9, 9], [0, 9], [2, 2]) == 3
""",
        },
        {
            "name": "Random vs brute force",
            "code": """
import random
sol = {fn}()
def brute(a, b, c):
    n = len(a); ans = 0
    for j in range(n):
        for i in range(j + 1):
            if a[i] == b[c[j] - 1]:
                ans += 1
    return ans
random.seed(1)
for _ in range(200):
    n = random.randint(1, 8)
    a = [random.randint(1, 5) for _ in range(n)]
    b = [random.randint(1, 5) for _ in range(n)]
    c = [random.randint(1, n) for _ in range(n)]
    assert sol.count_pairs(list(a), list(b), list(c)) == brute(a, b, c), (a, b, c)
""",
        },
    ],
}
