"""Switch-Controlled Lights — Baidu 2026 Autumn 7.23 (Problem 1).

Non-PyTorch algorithm problem, LeetCode class-mode.
n lights and n switches. Switch i (1-indexed) toggles lights i and i+1; switch
n toggles only light n. Given the initial on/off state a (0/1), return the
minimum number of switch presses to turn every light off.
"""

TASK = {
    "title": "Switch-Controlled Lights",
    "difficulty": "Easy",
    "function_name": "Solution",
    "hint": (
        "Sweep left to right. Light i can only still be toggled by switch i (switch i-1 is "
        "already fixed), so its final decision is forced: track the carry from the previous "
        "switch. cur = a[i] XOR prev; if cur == 1 you must press switch i (answer += 1, "
        "carry = 1), else carry = 0. Greedy is optimal and O(n)."
    ),
    "tests": [
        {
            "name": "Official samples",
            "code": """
sol = {fn}()
assert sol.min_ops([1, 1, 0, 1, 1]) == 2
assert sol.min_ops([1, 1, 1]) == 2
""",
        },
        {
            "name": "Already off / single light",
            "code": """
sol = {fn}()
assert sol.min_ops([0]) == 0
assert sol.min_ops([1]) == 1
assert sol.min_ops([0, 0, 0]) == 0
""",
        },
        {
            "name": "Random vs brute force",
            "code": """
import random, itertools
sol = {fn}()

def brute(a):
    n = len(a)
    best = None
    for combo in itertools.product([0, 1], repeat=n):
        light = a[:]
        for i in range(n):
            if combo[i]:
                light[i] ^= 1
                if i + 1 < n:
                    light[i + 1] ^= 1
        if all(v == 0 for v in light):
            s = sum(combo)
            best = s if best is None else min(best, s)
    return best

random.seed(1)
for _ in range(3000):
    n = random.randint(1, 11)
    a = [random.randint(0, 1) for _ in range(n)]
    assert sol.min_ops(list(a)) == brute(a), a
""",
        },
    ],
}
