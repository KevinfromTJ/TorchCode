"""Dynamic KV Cache Sparsification Manager — Huawei 2026-07-15 (Problem 1).

Non-PyTorch algorithm problem, LeetCode class-mode (stateful, like LRUCache).
"""

TASK = {
    "title": "Dynamic KV Cache Sparsification",
    "difficulty": "Medium",
    "function_name": "Solution",
    "hint": (
        "Construct with capacity K. Store tokens in a dict keyed by position -> (score, key, "
        "value). Keep a min-heap of (score, pos) to find the lowest-scoring token. On add: insert, "
        "push to heap; if size > K, pop the heap until the top position is still present (lazy "
        "deletion), delete it and return that position — heap order (score, then pos) makes ties "
        "evict the smaller position. On query: return items sorted by position ascending. "
        "add() returns the pruned position or None; query() returns a list of (pos, score, key, value)."
    ),
    "tests": [
        {
            "name": "Official sample sequence",
            "code": """
sol = {fn}(3)
assert sol.add(0, 5.0, [1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4]) is None
assert sol.add(1, 2.0, [2.0, 2.0, 2.0, 2.0], [0.5, 0.5, 0.5, 0.5]) is None
assert sol.add(2, 8.0, [3.0, 3.0, 3.0, 3.0], [0.9, 0.9, 0.9, 0.9]) is None
q = sol.query()
assert [t[0] for t in q] == [0, 1, 2]
assert q[0][1] == 5.0 and q[1][1] == 2.0 and q[2][1] == 8.0
assert list(q[0][2]) == [1.0, 2.0, 3.0, 4.0] and list(q[0][3]) == [0.1, 0.2, 0.3, 0.4]
assert sol.add(3, 6.0, [4.0, 4.0, 4.0, 4.0], [0.0, 0.0, 0.0, 0.0]) == 1   # evict lowest score (2.0)
assert [t[0] for t in sol.query()] == [0, 2, 3]
assert sol.add(4, 9.0, [5.0, 5.0, 5.0, 5.0], [1.0, 1.0, 1.0, 1.0]) == 0   # now lowest is pos 0 (5.0)
assert [t[0] for t in sol.query()] == [2, 3, 4]
""",
        },
        {
            "name": "Tie on score evicts smaller position",
            "code": """
sol = {fn}(2)
assert sol.add(0, 5.0, [0, 0, 0, 0], [0, 0, 0, 0]) is None
assert sol.add(1, 5.0, [0, 0, 0, 0], [0, 0, 0, 0]) is None
# all scores equal -> adding a third evicts the smallest position (0)
assert sol.add(2, 5.0, [0, 0, 0, 0], [0, 0, 0, 0]) == 0
assert [t[0] for t in sol.query()] == [1, 2]
""",
        },
        {
            "name": "Capacity one keeps only the newest survivor",
            "code": """
sol = {fn}(1)
assert sol.add(0, 3.0, [0, 0, 0, 0], [0, 0, 0, 0]) is None
# higher score arrives, but capacity 1 -> the lower of the two is evicted
assert sol.add(1, 9.0, [0, 0, 0, 0], [0, 0, 0, 0]) == 0
q = sol.query()
assert [t[0] for t in q] == [1] and q[0][1] == 9.0
# a lower score arrives -> it itself is the lowest and gets evicted immediately
assert sol.add(2, 1.0, [0, 0, 0, 0], [0, 0, 0, 0]) == 2
assert [t[0] for t in sol.query()] == [1]
""",
        },
        {
            "name": "No pruning below capacity",
            "code": """
sol = {fn}(5)
for p in range(4):
    assert sol.add(p, float(p + 1), [0, 0, 0, 0], [0, 0, 0, 0]) is None
assert len(sol.query()) == 4
""",
        },
    ],
}
