"""Bracket Layer Elimination — Baidu 2026 Autumn 7.30 (Problem 1).

Non-PyTorch algorithm problem, LeetCode class-mode.
Given a legal (balanced) bracket string, repeatedly remove ALL innermost pairs
"()" simultaneously in one round, until the string is empty. Return, for each
original character, the round number in which it disappears.
"""

TASK = {
    "title": "Bracket Layer Elimination",
    "difficulty": "Easy",
    "function_name": "Solution",
    "hint": (
        "The disappearance round is determined by nesting depth. Scan once: on '(' increase "
        "depth then record it; on ')' record depth then decrease. Let D be the max recorded "
        "depth. A character at depth d disappears in round D - d + 1 (the deepest layer, "
        "d = D, goes first)."
    ),
    "tests": [
        {
            "name": "Official samples",
            "code": """
sol = {fn}()
assert sol.rounds("(()())") == [2, 1, 1, 1, 1, 2]
assert sol.rounds("((()))") == [3, 2, 1, 1, 2, 3]
assert sol.rounds("(()((())))") == [4, 3, 3, 3, 2, 1, 1, 2, 3, 4]
""",
        },
        {
            "name": "Smallest cases",
            "code": """
sol = {fn}()
assert sol.rounds("()") == [1, 1]
assert sol.rounds("()()") == [1, 1, 1, 1]
""",
        },
        {
            "name": "Random vs simulation brute force",
            "code": """
import random
sol = {fn}()

def brute(s):
    n = len(s)
    ans = [0] * n
    cur = [(i, s[i]) for i in range(n)]
    r = 0
    while cur:
        r += 1
        depth = 0; d = []
        for _, ch in cur:
            if ch == '(':
                depth += 1; d.append(depth)
            else:
                d.append(depth); depth -= 1
        mx = max(d)
        rm = set(); i = 0
        while i < len(cur) - 1:
            if cur[i][1] == '(' and cur[i + 1][1] == ')' and d[i] == mx:
                rm.add(i); rm.add(i + 1); i += 2
            else:
                i += 1
        for idx in rm:
            ans[cur[idx][0]] = r
        cur = [cur[i] for i in range(len(cur)) if i not in rm]
    return ans

def gen(n):
    while True:
        depth = 0; s = []; ok = True
        for i in range(n):
            rem = n - i
            if depth == 0:
                c = '('
            elif depth == rem:
                c = ')'
            else:
                c = random.choice('()')
            depth += 1 if c == '(' else -1
            s.append(c)
            if depth < 0:
                ok = False; break
        if ok and depth == 0:
            return ''.join(s)

random.seed(3)
for _ in range(3000):
    n = random.randrange(2, 16, 2)
    s = gen(n)
    assert sol.rounds(s) == brute(s), s
""",
        },
    ],
}
