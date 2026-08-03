"""Shortest Path with One Free Edge — Dijkstra variant.

Non-PyTorch algorithm problem, LeetCode class-mode.
Directed weighted graph, positive weights. You may set the weight of at most one
edge to 0. Find the minimum-weight path from node 1 to node N, or -1 if none.
"""

TASK = {
    "title": "Shortest Path with One Free Edge",
    "difficulty": "Medium",
    "function_name": "Solution",
    "hint": (
        "Layered / state-augmented Dijkstra. State = (node, used) where used in {0,1} marks "
        "whether the one free edge has been spent. From (u,0) over edge (u,v,w): relax (v,0) "
        "with cost w (pay normally) AND relax (v,1) with cost 0 (spend the free edge here). "
        "From (u,1) over edge (u,v,w): only relax (v,1) with cost w. Run one Dijkstra over the "
        "2N states from (1,0). Answer = min(dist[N][0], dist[N][1]); return -1 if both are inf. "
        "Weights are positive so Dijkstra is valid."
    ),
    "tests": [
        {
            "name": "Basic examples",
            "code": """
sol = {fn}()
# 1 ->(5)-> 2 ->(3)-> 3 ; zeroing the 5-edge gives 3, zeroing the 3-edge gives 5 -> min 3
assert sol.min_weight_path(3, [(1, 2, 5), (2, 3, 3)]) == 3
# single edge 1->2 weight 7, N=2 -> zero it -> 0
assert sol.min_weight_path(2, [(1, 2, 7)]) == 0
# no path from 1 to N
assert sol.min_weight_path(3, [(1, 2, 4)]) == -1
""",
        },
        {
            "name": "Free edge should zero the largest edge on the best path",
            "code": """
sol = {fn}()
# path 1->2->3->4 with weights 1,100,1 ; zero the 100 -> total 2
assert sol.min_weight_path(4, [(1, 2, 1), (2, 3, 100), (3, 4, 1)]) == 2
# two routes: A) 1->4 weight 10 (zero->0) ; B) 1->2->4 weights 3,3 (zero one->3). min = 0
assert sol.min_weight_path(4, [(1, 4, 10), (1, 2, 3), (2, 4, 3)]) == 0
""",
        },
        {
            "name": "Trivial and self-loop / start==end cases",
            "code": """
sol = {fn}()
assert sol.min_weight_path(1, []) == 0                 # start == end, empty path
assert sol.min_weight_path(1, [(1, 1, 5)]) == 0        # start == end, self loop irrelevant
assert sol.min_weight_path(2, []) == -1                # disconnected
""",
        },
        {
            "name": "Choosing the free edge changes which route wins",
            "code": """
sol = {fn}()
# route A: 1->3 direct weight 8  -> zero -> 0
# route B: 1->2->3 weights 2,2   -> zero one -> 2
edges = [(1, 3, 8), (1, 2, 2), (2, 3, 2)]
assert sol.min_weight_path(3, edges) == 0
# now make direct route small so free edge is better spent on route B's big edge
# route A: 1->3 weight 5 (zero->0)  vs  route B: 1->2->3 weights 1,50 (zero 50 ->1)
edges2 = [(1, 3, 5), (1, 2, 1), (2, 3, 50)]
assert sol.min_weight_path(3, edges2) == 0
""",
        },
        {
            "name": "Parallel edges and cycles",
            "code": """
sol = {fn}()
# parallel edges 1->2 (weights 10 and 3); cycle 2->2 via 3 should not help
edges = [(1, 2, 10), (1, 2, 3), (2, 3, 4), (3, 2, 1), (2, 4, 6)]
# best without free: 3 + 6 = 9 ; free edge zeroes the 6 -> 3 ; or zeroes the 3 -> 6 ; min 3
assert sol.min_weight_path(4, edges) == 3
""",
        },
        {
            "name": "Random vs brute force (zero each edge, plain Dijkstra)",
            "code": """
import heapq, random

sol = {fn}()

def dijkstra(n, adj, src):
    INF = float('inf')
    dist = [INF] * (n + 1)
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(pq, (dist[v], v))
    return dist

def brute(n, edges):
    best = float('inf')
    # try zeroing edge index e, and also the "no free edge" case (e == -1)
    for e in range(-1, len(edges)):
        adj = [[] for _ in range(n + 1)]
        for i, (u, v, w) in enumerate(edges):
            adj[u].append((v, 0 if i == e else w))
        best = min(best, dijkstra(n, adj, 1)[n])
    return -1 if best == float('inf') else best

random.seed(1)
for _ in range(200):
    n = random.randint(1, 7)
    m = random.randint(0, 12)
    edges = [(random.randint(1, n), random.randint(1, n), random.randint(1, 20)) for _ in range(m)]
    assert sol.min_weight_path(n, list(edges)) == brute(n, edges), (n, edges)
""",
        },
    ],
}
