"""面试手撕代码（06）：图 —— BFS / DFS / 拓扑排序 / 并查集 / 最短路 —— 配套解法与测试。

https://arganzheng.life/coding-interview-graph-bfs-dfs-topological-union-find.html
"""
from __future__ import annotations

import heapq
import unittest
from collections import defaultdict, deque
from typing import Dict, List, Optional

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def num_islands(grid: List[List[str]]) -> int:
    """LC 200. 网格 DFS，访问过的格子直接改成 '0'（沉岛）。O(mn) / O(mn) 栈。"""
    m, n = len(grid), len(grid[0])

    def sink(i, j):
        if not (0 <= i < m and 0 <= j < n) or grid[i][j] != "1":
            return
        grid[i][j] = "0"
        for di, dj in DIRS:
            sink(i + di, j + dj)

    count = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == "1":
                count += 1
                sink(i, j)
    return count


def max_area_of_island(grid: List[List[int]]) -> int:
    """LC 695. 同一模板，DFS 返回面积。"""
    m, n = len(grid), len(grid[0])

    def area(i, j):
        if not (0 <= i < m and 0 <= j < n) or grid[i][j] != 1:
            return 0
        grid[i][j] = 0
        return 1 + sum(area(i + di, j + dj) for di, dj in DIRS)

    return max((area(i, j) for i in range(m) for j in range(n)), default=0)


def oranges_rotting(grid: List[List[int]]) -> int:
    """LC 994. 多源 BFS：所有腐烂橘子同时入队，按层计分钟。O(mn) / O(mn)。"""
    m, n = len(grid), len(grid[0])
    q = deque()
    fresh = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 2:
                q.append((i, j))
            elif grid[i][j] == 1:
                fresh += 1
    minutes = 0
    while q and fresh:
        for _ in range(len(q)):
            i, j = q.popleft()
            for di, dj in DIRS:
                x, y = i + di, j + dj
                if 0 <= x < m and 0 <= y < n and grid[x][y] == 1:
                    grid[x][y] = 2
                    fresh -= 1
                    q.append((x, y))
        minutes += 1
    return -1 if fresh else minutes


def can_finish(num_courses: int, prerequisites: List[List[int]]) -> bool:
    """LC 207. Kahn 拓扑：入度为 0 的进队，出队时给后继减入度。O(V+E)。"""
    return len(find_order(num_courses, prerequisites)) == num_courses


def find_order(num_courses: int, prerequisites: List[List[int]]) -> List[int]:
    """LC 210. 同上，返回顺序；有环则返回 []。"""
    graph = defaultdict(list)
    indeg = [0] * num_courses
    for a, b in prerequisites:              # b -> a
        graph[b].append(a)
        indeg[a] += 1
    q = deque(i for i in range(num_courses) if indeg[i] == 0)
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in graph[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order if len(order) == num_courses else []


def can_finish_dfs(num_courses: int, prerequisites: List[List[int]]) -> bool:
    """LC 207 的 DFS 三色标记版：灰色再遇到即有环。"""
    graph = defaultdict(list)
    for a, b in prerequisites:
        graph[b].append(a)
    color = [0] * num_courses               # 0 白 1 灰 2 黑

    def dfs(u):
        color[u] = 1
        for v in graph[u]:
            if color[v] == 1 or (color[v] == 0 and not dfs(v)):
                return False
        color[u] = 2
        return True

    return all(color[i] or dfs(i) for i in range(num_courses))


class GraphNode:
    def __init__(self, val: int = 0, neighbors: Optional[List["GraphNode"]] = None):
        self.val, self.neighbors = val, neighbors or []


def clone_graph(node: Optional[GraphNode]) -> Optional[GraphNode]:
    """LC 133. 哈希 old -> new 兼作 visited，DFS。O(V+E)。"""
    if not node:
        return None
    mapping: Dict[GraphNode, GraphNode] = {}

    def dfs(u):
        if u in mapping:
            return mapping[u]
        copy = GraphNode(u.val)
        mapping[u] = copy                    # 先登记再递归，处理环
        copy.neighbors = [dfs(v) for v in u.neighbors]
        return copy

    return dfs(node)


def ladder_length(begin: str, end: str, word_list: List[str]) -> int:
    """LC 127. 双向 BFS：每次扩展较小的一侧。O(N * L * 26)。"""
    words = set(word_list)
    if end not in words:
        return 0
    front, back = {begin}, {end}
    steps = 1
    while front and back:
        if len(front) > len(back):
            front, back = back, front
        nxt = set()
        for w in front:
            for i in range(len(w)):
                for c in "abcdefghijklmnopqrstuvwxyz":
                    cand = w[:i] + c + w[i + 1:]
                    if cand in back:
                        return steps + 1
                    if cand in words:
                        words.remove(cand)   # 用过就删，兼作 visited
                        nxt.add(cand)
        front = nxt
        steps += 1
    return 0


class UnionFind:
    """路径压缩 + 按大小合并；find 均摊近 O(1)。"""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n
        self.count = n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # 路径减半
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.count -= 1
        return True


def find_circle_num(is_connected: List[List[int]]) -> int:
    """LC 547. 并查集数连通块。O(n^2 α)。"""
    n = len(is_connected)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if is_connected[i][j]:
                uf.union(i, j)
    return uf.count


def accounts_merge(accounts: List[List[str]]) -> List[List[str]]:
    """LC 721. 邮箱映射到账户下标，同账户邮箱合并；输出按根聚合并排序。"""
    uf = UnionFind(len(accounts))
    owner: Dict[str, int] = {}
    for i, acc in enumerate(accounts):
        for email in acc[1:]:
            if email in owner:
                uf.union(i, owner[email])
            else:
                owner[email] = i
    groups = defaultdict(list)
    for email, i in owner.items():
        groups[uf.find(i)].append(email)
    return [[accounts[r][0]] + sorted(emails) for r, emails in groups.items()]


def network_delay_time(times: List[List[int]], n: int, k: int) -> int:
    """LC 743. Dijkstra（堆 + 懒删除）。O(E log E)。"""
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))
    dist = {k: 0}
    heap = [(0, k)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue                         # 过期条目
        for v, w in graph[u]:
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return max(dist.values()) if len(dist) == n else -1


def shortest_path_binary_matrix(grid: List[List[int]]) -> int:
    """LC 1091. 八方向网格 BFS。"""
    n = len(grid)
    if grid[0][0] or grid[-1][-1]:
        return -1
    q = deque([(0, 0, 1)])
    grid[0][0] = 1
    while q:
        i, j, d = q.popleft()
        if i == j == n - 1:
            return d
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                x, y = i + di, j + dj
                if 0 <= x < n and 0 <= y < n and grid[x][y] == 0:
                    grid[x][y] = 1
                    q.append((x, y, d + 1))
    return -1


class Tests(unittest.TestCase):
    def test_islands(self):
        g = [list("11000"), list("11000"), list("00100"), list("00011")]
        self.assertEqual(num_islands(g), 3)
        self.assertEqual(max_area_of_island([[0, 1, 1, 0], [0, 1, 0, 0], [1, 0, 0, 1]]), 3)

    def test_oranges(self):
        self.assertEqual(oranges_rotting([[2, 1, 1], [1, 1, 0], [0, 1, 1]]), 4)
        self.assertEqual(oranges_rotting([[2, 1, 1], [0, 1, 1], [1, 0, 1]]), -1)
        self.assertEqual(oranges_rotting([[0, 2]]), 0)

    def test_topo(self):
        self.assertTrue(can_finish(2, [[1, 0]]))
        self.assertFalse(can_finish(2, [[1, 0], [0, 1]]))
        self.assertEqual(find_order(4, [[1, 0], [2, 0], [3, 1], [3, 2]]), [0, 1, 2, 3])
        self.assertTrue(can_finish_dfs(3, [[1, 0], [2, 1]]))
        self.assertFalse(can_finish_dfs(3, [[1, 0], [2, 1], [0, 2]]))

    def test_clone(self):
        a, b, c = GraphNode(1), GraphNode(2), GraphNode(3)
        a.neighbors, b.neighbors, c.neighbors = [b, c], [a, c], [a, b]
        a2 = clone_graph(a)
        self.assertIsNot(a2, a)
        self.assertEqual(sorted(x.val for x in a2.neighbors), [2, 3])
        self.assertIs(a2.neighbors[0].neighbors[0], a2)

    def test_ladder(self):
        self.assertEqual(ladder_length("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]), 5)
        self.assertEqual(ladder_length("hit", "cog", ["hot", "dot", "dog", "lot", "log"]), 0)

    def test_union_find(self):
        self.assertEqual(find_circle_num([[1, 1, 0], [1, 1, 0], [0, 0, 1]]), 2)
        out = accounts_merge([["John", "johnsmith@mail.com", "john_newyork@mail.com"],
                              ["John", "johnsmith@mail.com", "john00@mail.com"],
                              ["Mary", "mary@mail.com"], ["John", "johnnybravo@mail.com"]])
        self.assertEqual(sorted(out), sorted([["John", "john00@mail.com", "john_newyork@mail.com", "johnsmith@mail.com"],
                                               ["Mary", "mary@mail.com"], ["John", "johnnybravo@mail.com"]]))

    def test_dijkstra(self):
        self.assertEqual(network_delay_time([[2, 1, 1], [2, 3, 1], [3, 4, 1]], 4, 2), 2)
        self.assertEqual(network_delay_time([[1, 2, 1]], 2, 2), -1)

    def test_binary_matrix(self):
        self.assertEqual(shortest_path_binary_matrix([[0, 0, 0], [1, 1, 0], [1, 1, 0]]), 4)
        self.assertEqual(shortest_path_binary_matrix([[1, 0, 0], [1, 1, 0], [1, 1, 0]]), -1)


if __name__ == "__main__":
    unittest.main(verbosity=1)
