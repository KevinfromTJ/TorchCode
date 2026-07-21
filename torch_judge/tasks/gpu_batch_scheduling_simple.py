"""GPU Batch Scheduling (No Discards) — simplified variant of Problem 2.

Non-PyTorch algorithm problem, LeetCode class-mode. Every request must be served
(no dropping), which collapses the DP into a single greedy sweep.
"""

TASK = {
    "title": "GPU Batch Scheduling (No Discards)",
    "difficulty": "Medium",
    "function_name": "Solution",
    "hint": (
        "No requests may be dropped, so just partition the sorted lengths into the fewest "
        "contiguous groups where each group has max-min <= K and size <= C. Sort ascending, "
        "then greedily grow each batch from the leftmost unbatched request: keep adding while "
        "a[i] - batch_start <= K and the batch holds fewer than C requests; start a new batch "
        "when either limit is hit. The greedy (extend maximally) is optimal for minimizing "
        "the number of contiguous groups."
    ),
    "tests": [
        {
            "name": "Basic cases",
            "code": """
sol = {fn}()
assert sol.min_batches([10, 11, 15, 16, 17], 1, 2) == 3
assert sol.min_batches([1, 100, 2, 200], 10, 3) == 3
""",
        },
        {
            "name": "Order independence (unsorted input)",
            "code": """
sol = {fn}()
assert sol.min_batches([17, 10, 16, 11, 15], 1, 2) == 3
assert sol.min_batches([200, 1, 100, 2], 10, 3) == 3
""",
        },
        {
            "name": "K=0 requires equal lengths per batch",
            "code": """
sol = {fn}()
assert sol.min_batches([1, 2, 3, 4], 0, 4) == 4       # all distinct -> 4 batches
assert sol.min_batches([1, 1, 1, 1, 1], 0, 2) == 3    # equal lengths, capacity 2 -> ceil(5/2)
assert sol.min_batches([1, 2, 3, 4, 5], 0, 1) == 5    # capacity 1 -> one per batch
""",
        },
        {
            "name": "Capacity C dominates when K is large",
            "code": """
sol = {fn}()
assert sol.min_batches([1, 2, 3, 4, 5, 6], 100, 3) == 2   # ceil(6/3)
assert sol.min_batches([5, 5, 5, 5], 1000000000, 10) == 1
""",
        },
        {
            "name": "Single request",
            "code": """
sol = {fn}()
assert sol.min_batches([5], 0, 1) == 1
assert sol.min_batches([42], 1000000000, 5) == 1
""",
        },
        {
            "name": "Spread wider than K forces separate batches",
            "code": """
sol = {fn}()
assert sol.min_batches([10, 20, 30], 5, 3) == 3
# window resets each time the running max-min would exceed K
assert sol.min_batches([1, 2, 3, 10, 11, 12], 2, 10) == 2
""",
        },
        {
            "name": "Every request is served (batches * capacity covers N)",
            "code": """
import random
sol = {fn}()
random.seed(0)
for _ in range(200):
    n = random.randint(1, 40)
    lengths = [random.randint(1, 30) for _ in range(n)]
    k = random.randint(0, 10)
    c = random.randint(1, n)
    b = sol.min_batches(list(lengths), k, c)
    assert b >= (n + c - 1) // c, 'cannot use fewer batches than capacity allows'
    assert b <= n, 'never need more than one batch per request'
""",
        },
    ],
}
