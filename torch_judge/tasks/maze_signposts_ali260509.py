"""Maze Signposts — Alibaba 2026-05-09 (Problem 2).

Non-PyTorch algorithm problem, LeetCode class-mode.
Corridor of positions 1..n, each with a sign 'L' or 'R'. A ball dropped at
start s repeatedly reads the sign at its current cell and jumps in that
direction to the first not-yet-visited position; it exits at 0 (off the left)
or n+1 (off the right). For every start s = 1..n, report the exit side.

Key insight: exactly cntL = (number of 'L' signs) balls exit left, and they
are the ones dropped at the leftmost cntL starts. So the answer string is
'L' * cntL + 'R' * (n - cntL).
"""

TASK = {
    "title": "Maze Signposts",
    "difficulty": "Medium",
    "function_name": "Solution",
    "hint": (
        "Simulating each start is O(n^2). The clean observation: the number of balls that "
        "exit on the left equals the number of 'L' signs (cntL), and those are exactly the "
        "starts 1..cntL. So the result is 'L' repeated cntL times followed by 'R' for the "
        "rest. Return 'L'*cntL + 'R'*(n-cntL)."
    ),
    "tests": [
        {
            "name": "Official samples",
            "code": """
sol = {fn}()
assert sol.exit_sides("LRRLL") == "LLLRR"
assert sol.exit_sides("LLL") == "LLL"
""",
        },
        {
            "name": "Edge single / all same",
            "code": """
sol = {fn}()
assert sol.exit_sides("L") == "L"
assert sol.exit_sides("R") == "R"
assert sol.exit_sides("RRRR") == "RRRR"
assert sol.exit_sides("LLLL") == "LLLL"
""",
        },
        {
            "name": "Random vs direct simulation",
            "code": """
import random
sol = {fn}()

def brute(signs):
    n = len(signs); out = []
    for s in range(1, n + 1):
        visited = [False] * (n + 2)
        cur = s; visited[cur] = True
        while True:
            if signs[cur - 1] == 'L':
                j = cur - 1
                while j >= 1 and visited[j]:
                    j -= 1
                if j == 0:
                    out.append('L'); break
                cur = j; visited[j] = True
            else:
                j = cur + 1
                while j <= n and visited[j]:
                    j += 1
                if j == n + 1:
                    out.append('R'); break
                cur = j; visited[j] = True
    return ''.join(out)

random.seed(7)
for _ in range(2000):
    n = random.randint(1, 12)
    signs = ''.join(random.choice('LR') for _ in range(n))
    assert sol.exit_sides(signs) == brute(signs), signs
""",
        },
    ],
}
