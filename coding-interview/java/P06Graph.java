import java.util.*;

/** 面试手撕代码（06）：图 —— BFS / DFS / 拓扑 / 并查集 / 最短路。运行：java -ea P06Graph */
public class P06Graph {

    static final int[][] DIRS = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    /** LC 200. 沉岛 DFS。 */
    static int numIslands(char[][] g) {
        int count = 0;
        for (int i = 0; i < g.length; i++)
            for (int j = 0; j < g[0].length; j++)
                if (g[i][j] == '1') {
                    count++;
                    sink(g, i, j);
                }
        return count;
    }

    private static void sink(char[][] g, int i, int j) {
        if (i < 0 || j < 0 || i >= g.length || j >= g[0].length || g[i][j] != '1') return;
        g[i][j] = '0';
        for (int[] d : DIRS) sink(g, i + d[0], j + d[1]);
    }

    /** LC 994. 多源 BFS 按层计时。 */
    static int orangesRotting(int[][] g) {
        int m = g.length, n = g[0].length, fresh = 0, minutes = 0;
        Deque<int[]> q = new ArrayDeque<>();
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) {
                if (g[i][j] == 2) q.add(new int[] {i, j});
                else if (g[i][j] == 1) fresh++;
            }
        while (!q.isEmpty() && fresh > 0) {
            for (int s = q.size(); s > 0; s--) {
                int[] c = q.poll();
                for (int[] d : DIRS) {
                    int x = c[0] + d[0], y = c[1] + d[1];
                    if (x >= 0 && y >= 0 && x < m && y < n && g[x][y] == 1) {
                        g[x][y] = 2;
                        fresh--;
                        q.add(new int[] {x, y});
                    }
                }
            }
            minutes++;
        }
        return fresh == 0 ? minutes : -1;
    }

    /** LC 210（含 207）. Kahn 拓扑排序，有环返回空数组。 */
    static int[] findOrder(int numCourses, int[][] prerequisites) {
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) graph.add(new ArrayList<>());
        int[] indeg = new int[numCourses];
        for (int[] p : prerequisites) {
            graph.get(p[1]).add(p[0]);
            indeg[p[0]]++;
        }
        Deque<Integer> q = new ArrayDeque<>();
        for (int i = 0; i < numCourses; i++) if (indeg[i] == 0) q.add(i);
        int[] order = new int[numCourses];
        int k = 0;
        while (!q.isEmpty()) {
            int u = q.poll();
            order[k++] = u;
            for (int v : graph.get(u)) if (--indeg[v] == 0) q.add(v);
        }
        return k == numCourses ? order : new int[0];
    }

    static boolean canFinish(int n, int[][] pre) {
        return findOrder(n, pre).length == n;
    }

    static class GraphNode {
        int val;
        List<GraphNode> neighbors = new ArrayList<>();

        GraphNode(int v) {
            val = v;
        }
    }

    /** LC 133. 哈希兼作 visited 的 DFS。 */
    static GraphNode cloneGraph(GraphNode node) {
        return node == null ? null : clone(node, new HashMap<>());
    }

    private static GraphNode clone(GraphNode u, Map<GraphNode, GraphNode> map) {
        if (map.containsKey(u)) return map.get(u);
        GraphNode copy = new GraphNode(u.val);
        map.put(u, copy);
        for (GraphNode v : u.neighbors) copy.neighbors.add(clone(v, map));
        return copy;
    }

    /** LC 127. 双向 BFS。 */
    static int ladderLength(String begin, String end, List<String> wordList) {
        Set<String> words = new HashSet<>(wordList);
        if (!words.contains(end)) return 0;
        Set<String> front = new HashSet<>(List.of(begin)), back = new HashSet<>(List.of(end));
        int steps = 1;
        while (!front.isEmpty() && !back.isEmpty()) {
            if (front.size() > back.size()) {
                Set<String> t = front;
                front = back;
                back = t;
            }
            Set<String> next = new HashSet<>();
            for (String w : front) {
                char[] cs = w.toCharArray();
                for (int i = 0; i < cs.length; i++) {
                    char orig = cs[i];
                    for (char c = 'a'; c <= 'z'; c++) {
                        cs[i] = c;
                        String cand = new String(cs);
                        if (back.contains(cand)) return steps + 1;
                        if (words.remove(cand)) next.add(cand);
                    }
                    cs[i] = orig;
                }
            }
            front = next;
            steps++;
        }
        return 0;
    }

    /** 并查集：路径压缩 + 按大小合并。 */
    static class UnionFind {
        int[] parent, size;
        int count;

        UnionFind(int n) {
            parent = new int[n];
            size = new int[n];
            count = n;
            for (int i = 0; i < n; i++) {
                parent[i] = i;
                size[i] = 1;
            }
        }

        int find(int x) {
            while (parent[x] != x) {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            return x;
        }

        boolean union(int a, int b) {
            int ra = find(a), rb = find(b);
            if (ra == rb) return false;
            if (size[ra] < size[rb]) {
                int t = ra;
                ra = rb;
                rb = t;
            }
            parent[rb] = ra;
            size[ra] += size[rb];
            count--;
            return true;
        }
    }

    /** LC 547. */
    static int findCircleNum(int[][] isConnected) {
        int n = isConnected.length;
        UnionFind uf = new UnionFind(n);
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) if (isConnected[i][j] == 1) uf.union(i, j);
        return uf.count;
    }

    /** LC 721. */
    static List<List<String>> accountsMerge(List<List<String>> accounts) {
        UnionFind uf = new UnionFind(accounts.size());
        Map<String, Integer> owner = new HashMap<>();
        for (int i = 0; i < accounts.size(); i++)
            for (String email : accounts.get(i).subList(1, accounts.get(i).size())) {
                Integer j = owner.putIfAbsent(email, i);
                if (j != null) uf.union(i, j);
            }
        Map<Integer, TreeSet<String>> groups = new HashMap<>();
        for (Map.Entry<String, Integer> e : owner.entrySet())
            groups.computeIfAbsent(uf.find(e.getValue()), k -> new TreeSet<>()).add(e.getKey());
        List<List<String>> out = new ArrayList<>();
        for (Map.Entry<Integer, TreeSet<String>> e : groups.entrySet()) {
            List<String> acc = new ArrayList<>();
            acc.add(accounts.get(e.getKey()).get(0));
            acc.addAll(e.getValue());
            out.add(acc);
        }
        return out;
    }

    /** LC 743. Dijkstra（堆 + 懒删除）。 */
    static int networkDelayTime(int[][] times, int n, int k) {
        List<List<int[]>> graph = new ArrayList<>();
        for (int i = 0; i <= n; i++) graph.add(new ArrayList<>());
        for (int[] t : times) graph.get(t[0]).add(new int[] {t[1], t[2]});
        int[] dist = new int[n + 1];
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[k] = 0;
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> Integer.compare(a[0], b[0]));
        pq.offer(new int[] {0, k});
        while (!pq.isEmpty()) {
            int[] cur = pq.poll();
            int d = cur[0], u = cur[1];
            if (d > dist[u]) continue;
            for (int[] e : graph.get(u)) {
                int nd = d + e[1];
                if (nd < dist[e[0]]) {
                    dist[e[0]] = nd;
                    pq.offer(new int[] {nd, e[0]});
                }
            }
        }
        int best = 0;
        for (int i = 1; i <= n; i++) {
            if (dist[i] == Integer.MAX_VALUE) return -1;
            best = Math.max(best, dist[i]);
        }
        return best;
    }

    public static void main(String[] args) {
        char[][] g = {
            "11000".toCharArray(),
            "11000".toCharArray(),
            "00100".toCharArray(),
            "00011".toCharArray()
        };
        assert numIslands(g) == 3;
        assert orangesRotting(new int[][] {{2, 1, 1}, {1, 1, 0}, {0, 1, 1}}) == 4;
        assert orangesRotting(new int[][] {{2, 1, 1}, {0, 1, 1}, {1, 0, 1}}) == -1;
        assert canFinish(2, new int[][] {{1, 0}}) && !canFinish(2, new int[][] {{1, 0}, {0, 1}});
        assert Arrays.equals(
                findOrder(4, new int[][] {{1, 0}, {2, 0}, {3, 1}, {3, 2}}), new int[] {0, 1, 2, 3});
        GraphNode a = new GraphNode(1), b = new GraphNode(2);
        a.neighbors.add(b);
        b.neighbors.add(a);
        GraphNode a2 = cloneGraph(a);
        assert a2 != a && a2.neighbors.get(0).neighbors.get(0) == a2;
        assert ladderLength("hit", "cog", List.of("hot", "dot", "dog", "lot", "log", "cog")) == 5;
        assert findCircleNum(new int[][] {{1, 1, 0}, {1, 1, 0}, {0, 0, 1}}) == 2;
        List<List<String>> merged =
                accountsMerge(
                        List.of(
                                List.of("John", "johnsmith@mail.com", "john_newyork@mail.com"),
                                List.of("John", "johnsmith@mail.com", "john00@mail.com"),
                                List.of("Mary", "mary@mail.com"),
                                List.of("John", "johnnybravo@mail.com")));
        assert merged.size() == 3;
        assert networkDelayTime(new int[][] {{2, 1, 1}, {2, 3, 1}, {3, 4, 1}}, 4, 2) == 2;
        assert networkDelayTime(new int[][] {{1, 2, 1}}, 2, 2) == -1;
        System.out.println("P06Graph OK");
    }
}
