"""Single-Cycle Graph Alarm-Flip Maintenance — Pinduoduo 2026-07-19 (Problem 4).

Non-PyTorch algorithm problem, LeetCode class-mode.
"""

TASK = {
    "title": "Single-Cycle Graph Alarm Flip",
    "difficulty": "Hard",
    "function_name": "Solution",
    "hint": (
        "Maintaining an edge flips both endpoints, so each node needs (XOR of its chosen incident "
        "edges) == b_v. XOR-ing all node equations, every edge cancels, so a solution exists iff "
        "XOR of all b_v == 0 -> otherwise return [-1, 0]. The graph is connected with N nodes and "
        "N edges = spanning tree + one extra edge (exactly one cycle). Build a spanning tree, then "
        "process leaves-to-root: if a node still needs flipping, select its parent edge (flipping "
        "the parent). This gives one feasible plan. The ONLY other plan toggles every edge on the "
        "unique cycle (parity unchanged). Compare the two costs: return [min_cost, 1] if they "
        "differ, else [equal_cost, 2]."
    ),
    "tests": [
        {
            "name": "Official samples",
            "code": """
sol = {fn}()
assert list(sol.solve(5, [0, 1, 1, 1, 1], [[1, 2, 4], [2, 3, 2], [3, 1, 7], [3, 4, 5], [4, 5, 1]])) == [3, 1]
assert list(sol.solve(4, [1, 1, 1, 1], [[1, 2, 1], [2, 3, 1], [3, 4, 1], [4, 1, 1]])) == [2, 2]
assert list(sol.solve(3, [1, 0, 0], [[1, 2, 1], [2, 3, 1], [3, 1, 1]])) == [-1, 0]
""",
        },
        {
            "name": "Infeasible when total parity is odd",
            "code": """
sol = {fn}()
assert list(sol.solve(3, [1, 1, 1], [[1, 2, 5], [2, 3, 3], [3, 1, 4]])) == [-1, 0]
assert list(sol.solve(4, [1, 1, 1, 0], [[1, 2, 1], [2, 3, 1], [3, 4, 1], [4, 1, 1]])) == [-1, 0]
""",
        },
        {
            "name": "All alarms already off -> empty plan",
            "code": """
sol = {fn}()
# doing nothing costs 0; flipping the whole cycle costs more -> unique
assert list(sol.solve(3, [0, 0, 0], [[1, 2, 1], [2, 3, 1], [3, 1, 1]])) == [0, 1]
assert list(sol.solve(4, [0, 0, 0, 0], [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 1, 6]])) == [0, 1]
""",
        },
        {
            "name": "Two equal-cost plans give count 2",
            "code": """
sol = {fn}()
# unit-cost 4-cycle: the two feasible plans cost the same
assert list(sol.solve(4, [1, 1, 1, 1], [[1, 2, 1], [2, 3, 1], [3, 4, 1], [4, 1, 1]])) == [2, 2]
""",
        },
        {
            "name": "Cheaper plan is unique (costs differ)",
            "code": """
sol = {fn}()
assert list(sol.solve(4, [1, 0, 1, 0], [[1, 2, 2], [2, 3, 3], [3, 4, 1], [4, 1, 5]])) == [5, 1]
""",
        },
        {
            "name": "Tree branches hanging off the cycle",
            "code": """
sol = {fn}()
# triangle (1-2-3) with tails 4 and 5; only nodes 4,5,and their parents need flips
res = sol.solve(5, [0, 1, 1, 1, 1], [[1, 2, 4], [2, 3, 2], [3, 1, 7], [3, 4, 5], [4, 5, 1]])
assert list(res) == [3, 1]
""",
        },
    ],
}
