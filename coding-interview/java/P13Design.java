import java.util.*;

/** 面试手撕代码（13）：设计题与数据结构实现。运行：java -ea P13Design */
public class P13Design {

    /** LC 146. 哈希 + 双向链表（哨兵）。 */
    static class LRUCache {
        private static class Node {
            int key, val;
            Node prev, next;

            Node(int k, int v) {
                key = k;
                val = v;
            }
        }

        private final int cap;
        private final Map<Integer, Node> map = new HashMap<>();
        private final Node head = new Node(0, 0), tail = new Node(0, 0);

        LRUCache(int capacity) {
            cap = capacity;
            head.next = tail;
            tail.prev = head;
        }

        private void remove(Node n) {
            n.prev.next = n.next;
            n.next.prev = n.prev;
        }

        private void addFront(Node n) {
            n.next = head.next;
            n.prev = head;
            head.next.prev = n;
            head.next = n;
        }

        int get(int key) {
            Node n = map.get(key);
            if (n == null) return -1;
            remove(n);
            addFront(n);
            return n.val;
        }

        void put(int key, int value) {
            Node n = map.get(key);
            if (n != null) {
                n.val = value;
                remove(n);
                addFront(n);
                return;
            }
            if (map.size() == cap) {
                Node lru = tail.prev;
                remove(lru);
                map.remove(lru.key);
            }
            n = new Node(key, value);
            map.put(key, n);
            addFront(n);
        }
    }

    /** LC 146 的 LinkedHashMap 版：accessOrder=true + removeEldestEntry。 */
    static class LRUCacheLinked extends LinkedHashMap<Integer, Integer> {
        private static final long serialVersionUID = 1L;
        private final int cap;

        LRUCacheLinked(int capacity) {
            super(16, 0.75f, true);
            cap = capacity;
        }

        int get(int key) {
            return super.getOrDefault(key, -1);
        }

        void put2(int key, int value) {
            super.put(key, value);
        }

        @Override
        protected boolean removeEldestEntry(Map.Entry<Integer, Integer> eldest) {
            return size() > cap;
        }
    }

    /** LC 460. freq -> LinkedHashSet（插入序 = LRU 序），minFreq 追踪。 */
    static class LFUCache {
        private final int cap;
        private final Map<Integer, Integer> kv = new HashMap<>(), kf = new HashMap<>();
        private final Map<Integer, LinkedHashSet<Integer>> buckets = new HashMap<>();
        private int minFreq = 0;

        LFUCache(int capacity) {
            cap = capacity;
        }

        private void touch(int key) {
            int f = kf.get(key);
            buckets.get(f).remove(key);
            if (buckets.get(f).isEmpty()) {
                buckets.remove(f);
                if (minFreq == f) minFreq = f + 1;
            }
            kf.put(key, f + 1);
            buckets.computeIfAbsent(f + 1, k -> new LinkedHashSet<>()).add(key);
        }

        int get(int key) {
            if (!kv.containsKey(key)) return -1;
            touch(key);
            return kv.get(key);
        }

        void put(int key, int value) {
            if (cap == 0) return;
            if (kv.containsKey(key)) {
                kv.put(key, value);
                touch(key);
                return;
            }
            if (kv.size() == cap) {
                LinkedHashSet<Integer> b = buckets.get(minFreq);
                int evict = b.iterator().next();
                b.remove(evict);
                if (b.isEmpty()) buckets.remove(minFreq);
                kv.remove(evict);
                kf.remove(evict);
            }
            kv.put(key, value);
            kf.put(key, 1);
            buckets.computeIfAbsent(1, k -> new LinkedHashSet<>()).add(key);
            minFreq = 1;
        }
    }

    /** LC 208. 26 叉数组版。 */
    static class Trie {
        private static class Node {
            Node[] ch = new Node[26];
            boolean end;
        }

        private final Node root = new Node();

        void insert(String w) {
            Node n = root;
            for (char c : w.toCharArray()) {
                int i = c - 'a';
                if (n.ch[i] == null) n.ch[i] = new Node();
                n = n.ch[i];
            }
            n.end = true;
        }

        private Node walk(String p) {
            Node n = root;
            for (char c : p.toCharArray()) {
                n = n.ch[c - 'a'];
                if (n == null) return null;
            }
            return n;
        }

        boolean search(String w) {
            Node n = walk(w);
            return n != null && n.end;
        }

        boolean startsWith(String p) {
            return walk(p) != null;
        }
    }

    /** LC 212. Trie + 网格 DFS，命中后置 end=false 去重。 */
    static List<String> findWords(char[][] board, String[] words) {
        TrieNode root = new TrieNode();
        for (String w : words) {
            TrieNode n = root;
            for (char c : w.toCharArray()) {
                if (n.ch[c - 'a'] == null) n.ch[c - 'a'] = new TrieNode();
                n = n.ch[c - 'a'];
            }
            n.word = w;
        }
        List<String> out = new ArrayList<>();
        for (int i = 0; i < board.length; i++)
            for (int j = 0; j < board[0].length; j++) dfs(board, i, j, root, out);
        return out;
    }

    private static class TrieNode {
        TrieNode[] ch = new TrieNode[26];
        String word;
    }

    private static void dfs(char[][] b, int i, int j, TrieNode parent, List<String> out) {
        if (i < 0 || j < 0 || i >= b.length || j >= b[0].length || b[i][j] == '#') return;
        char c = b[i][j];
        TrieNode n = parent.ch[c - 'a'];
        if (n == null) return;
        if (n.word != null) {
            out.add(n.word);
            n.word = null;
        }
        b[i][j] = '#';
        dfs(b, i + 1, j, n, out);
        dfs(b, i - 1, j, n, out);
        dfs(b, i, j + 1, n, out);
        dfs(b, i, j - 1, n, out);
        b[i][j] = c;
    }

    /** LC 380. 数组 + 下标哈希。 */
    static class RandomizedSet {
        private final List<Integer> vals = new ArrayList<>();
        private final Map<Integer, Integer> pos = new HashMap<>();
        private final Random rnd = new Random();

        boolean insert(int v) {
            if (pos.containsKey(v)) return false;
            pos.put(v, vals.size());
            vals.add(v);
            return true;
        }

        boolean remove(int v) {
            Integer i = pos.get(v);
            if (i == null) return false;
            int last = vals.get(vals.size() - 1);
            vals.set(i, last);
            pos.put(last, i);
            vals.remove(vals.size() - 1);
            pos.remove(v);
            return true;
        }

        int getRandom() {
            return vals.get(rnd.nextInt(vals.size()));
        }
    }

    /** LC 307. 树状数组。 */
    static class NumArray {
        private final int n;
        private final int[] nums, tree;

        NumArray(int[] a) {
            n = a.length;
            nums = a.clone();
            tree = new int[n + 1];
            for (int i = 0; i < n; i++) add(i + 1, a[i]);
        }

        private void add(int i, int d) {
            for (; i <= n; i += i & -i) tree[i] += d;
        }

        private int prefix(int i) {
            int s = 0;
            for (; i > 0; i -= i & -i) s += tree[i];
            return s;
        }

        void update(int index, int val) {
            add(index + 1, val - nums[index]);
            nums[index] = val;
        }

        int sumRange(int l, int r) {
            return prefix(r + 1) - prefix(l);
        }
    }

    /** LC 232. */
    static class MyQueue {
        private final Deque<Integer> in = new ArrayDeque<>(), out = new ArrayDeque<>();

        void push(int x) {
            in.push(x);
        }

        private void shift() {
            if (out.isEmpty()) while (!in.isEmpty()) out.push(in.pop());
        }

        int pop() {
            shift();
            return out.pop();
        }

        int peek() {
            shift();
            return out.peek();
        }

        boolean empty() {
            return in.isEmpty() && out.isEmpty();
        }
    }

    /** LC 981. 每个 key 一棵 TreeMap，floorEntry 即"<= timestamp 的最大"。 */
    static class TimeMap {
        private final Map<String, TreeMap<Integer, String>> store = new HashMap<>();

        void set(String k, String v, int t) {
            store.computeIfAbsent(k, x -> new TreeMap<>()).put(t, v);
        }

        String get(String k, int t) {
            TreeMap<Integer, String> m = store.get(k);
            if (m == null) return "";
            Map.Entry<Integer, String> e = m.floorEntry(t);
            return e == null ? "" : e.getValue();
        }
    }

    public static void main(String[] args) {
        LRUCache c = new LRUCache(2);
        c.put(1, 1);
        c.put(2, 2);
        assert c.get(1) == 1;
        c.put(3, 3);
        assert c.get(2) == -1;
        c.put(4, 4);
        assert c.get(1) == -1 && c.get(3) == 3 && c.get(4) == 4;
        LRUCacheLinked l = new LRUCacheLinked(2);
        l.put2(1, 1);
        l.put2(2, 2);
        l.get(1);
        l.put2(3, 3);
        assert l.get(2) == -1 && l.get(1) == 1;
        LFUCache f = new LFUCache(2);
        f.put(1, 1);
        f.put(2, 2);
        assert f.get(1) == 1;
        f.put(3, 3);
        assert f.get(2) == -1 && f.get(3) == 3;
        f.put(4, 4);
        assert f.get(1) == -1 && f.get(3) == 3 && f.get(4) == 4;
        Trie t = new Trie();
        t.insert("apple");
        assert t.search("apple") && !t.search("app") && t.startsWith("app");
        char[][] board = {
            "oaan".toCharArray(), "etae".toCharArray(), "ihkr".toCharArray(), "iflv".toCharArray()
        };
        List<String> found = findWords(board, new String[] {"oath", "pea", "eat", "rain"});
        Collections.sort(found);
        assert found.equals(List.of("eat", "oath"));
        RandomizedSet rs = new RandomizedSet();
        assert rs.insert(1) && !rs.remove(2) && rs.insert(2) && rs.remove(1) && rs.getRandom() == 2;
        NumArray na = new NumArray(new int[] {1, 3, 5});
        assert na.sumRange(0, 2) == 9;
        na.update(1, 2);
        assert na.sumRange(0, 2) == 8;
        MyQueue q = new MyQueue();
        q.push(1);
        q.push(2);
        assert q.peek() == 1 && q.pop() == 1 && !q.empty();
        TimeMap tm = new TimeMap();
        tm.set("foo", "bar", 1);
        tm.set("foo", "bar2", 4);
        assert tm.get("foo", 3).equals("bar")
                && tm.get("foo", 5).equals("bar2")
                && tm.get("foo", 0).isEmpty();
        System.out.println("P13Design OK");
    }
}
