"""Agent Execution Trajectory Compression — Huawei 2026-07-15 (Problem 2).

Non-PyTorch algorithm problem, LeetCode class-mode. TSP-style bitmask DP.
"""

TASK = {
    "title": "Agent Trajectory Compression",
    "difficulty": "Hard",
    "function_name": "Solution",
    "hint": (
        "Simulate the original trajectory once (each step: move, clamped at borders; then apply "
        "I/D/N to the current cell) to get the final net value of every cell and the final agent "
        "position (the required end). Only cells with non-zero net value must be visited (>=1 op "
        "each), and at most 12 such cells exist. Distances are Manhattan (no obstacles). Run a "
        "bitmask TSP DP: dp[mask][i] = min steps to start at center, cover set mask, end at cell "
        "i. Base dp[1<<i][i] = dist(center, i), EXCEPT if cell i is the center itself it costs 2 "
        "(you must leave and return, since you cannot operate at t=0). Answer = min over i of "
        "dp[full][i] + dist(i, end). If no non-zero cell, answer = dist(center, end)."
    ),
    "tests": [
        {
            "name": "Official sample 2",
            "code": """
sol = {fn}()
ops = [('R','I',2),('R','D',1),('U','I',3),('L','I',1),('D','D',2),('R','I',4)]
assert sol.min_steps(5, 5, ops) == 4
""",
        },
        {
            "name": "No effective operations -> just walk to end",
            "code": """
sol = {fn}()
# all N ops: grid stays zero, only the final position matters
assert sol.min_steps(3, 3, [('R','N',0), ('D','N',0)]) == 2     # center (1,1) -> end (2,2)
assert sol.min_steps(3, 3, [('U','N',0)]) == 1                  # center -> (0,1)
assert sol.min_steps(3, 3, [('U','N',0), ('D','N',0)]) == 0     # ends back at center
""",
        },
        {
            "name": "Single non-zero cell",
            "code": """
sol = {fn}()
# move R to (1,2), add 7 there; must visit it and it is also the end
assert sol.min_steps(3, 3, [('R','I',7)]) == 1
""",
        },
        {
            "name": "Center cell needs an op (must leave and return)",
            "code": """
sol = {fn}()
# R I5 -> (1,2)=5 ; L D5 -> back to center (1,1)=-5. Two non-zero cells, end is the center.
assert sol.min_steps(3, 3, [('R','I',5), ('L','D',5)]) == 2
""",
        },
        {
            "name": "Random vs permutation brute force",
            "code": """
import itertools, random
sol = {fn}()
def brute(n, m, ops):
    grid = [[0]*m for _ in range(n)]
    sr, sc = n//2, m//2; r, c = sr, sc
    for mv, op, val in ops:
        nr, nc = r, c
        if mv=='U': nr-=1
        elif mv=='D': nr+=1
        elif mv=='L': nc-=1
        else: nc+=1
        if 0<=nr<n and 0<=nc<m: r, c = nr, nc
        if op=='I': grid[r][c]+=val
        elif op=='D': grid[r][c]-=val
    er, ec = r, c
    pos = [(i,j) for i in range(n) for j in range(m) if grid[i][j]!=0]
    if not pos: return abs(sr-er)+abs(sc-ec)
    best = float('inf')
    for perm in itertools.permutations(pos):
        cost = 2 if perm[0]==(sr,sc) else abs(sr-perm[0][0])+abs(sc-perm[0][1])
        for a, b in zip(perm, perm[1:]):
            cost += abs(a[0]-b[0])+abs(a[1]-b[1])
        cost += abs(perm[-1][0]-er)+abs(perm[-1][1]-ec)
        best = min(best, cost)
    return best
random.seed(3)
for _ in range(120):
    n = random.choice([3, 5, 7]); m = random.choice([3, 5, 7]); K = random.randint(1, 8)
    ops = []
    for _ in range(K):
        op = random.choice('IDN')
        ops.append((random.choice('UDLR'), op, 0 if op == 'N' else random.randint(1, 5)))
    assert sol.min_steps(n, m, list(ops)) == brute(n, m, ops), (n, m, ops)
""",
        },
    ],
}
