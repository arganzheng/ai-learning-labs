"""面试手撕代码（13）：设计题与数据结构实现 —— 配套解法与测试。

https://arganzheng.life/coding-interview-design-problems-lru-lfu-trie.html
"""
from __future__ import annotations

import heapq
import random
import unittest
from bisect import bisect_left, bisect_right, insort
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional


# ---------- LC 146 LRU ----------

class _DNode:
    __slots__ = ("key", "val", "prev", "next")

    def __init__(self, key: int = 0, val: int = 0):
        self.key, self.val = key, val
        self.prev: Optional[_DNode] = None
        self.next: Optional[_DNode] = None


class LRUCache:
    """哈希表 + 双向链表（带头尾哨兵）：get / put 都 O(1)。最近使用的靠近 head。"""

    def __init__(self, capacity: int):
        self.cap = capacity
        self.map: Dict[int, _DNode] = {}
        self.head, self.tail = _DNode(), _DNode()
        self.head.next, self.tail.prev = self.tail, self.head

    def _remove(self, node: _DNode) -> None:
        node.prev.next, node.next.prev = node.next, node.prev

    def _add_front(self, node: _DNode) -> None:
        node.next, node.prev = self.head.next, self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        if key not in self.map:
            return -1
        node = self.map[key]
        self._remove(node)
        self._add_front(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.map:
            node = self.map[key]
            node.val = value
            self._remove(node)
            self._add_front(node)
            return
        if len(self.map) == self.cap:
            lru = self.tail.prev
            self._remove(lru)
            del self.map[lru.key]
        node = _DNode(key, value)
        self.map[key] = node
        self._add_front(node)


class LRUCacheOrdered:
    """同一题的 OrderedDict 版（面试里先说清手写版再提这个）。"""

    def __init__(self, capacity: int):
        self.cap = capacity
        self.od: "OrderedDict[int, int]" = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.od:
            return -1
        self.od.move_to_end(key)
        return self.od[key]

    def put(self, key: int, value: int) -> None:
        if key in self.od:
            self.od.move_to_end(key)
        self.od[key] = value
        if len(self.od) > self.cap:
            self.od.popitem(last=False)


# ---------- LC 460 LFU ----------

class LFUCache:
    """freq -> 该频次的 OrderedDict（按插入序即 LRU 序）；min_freq 追踪最小频次。全部 O(1)。"""

    def __init__(self, capacity: int):
        self.cap = capacity
        self.kv: Dict[int, int] = {}
        self.kf: Dict[int, int] = {}
        self.buckets: Dict[int, "OrderedDict[int, None]"] = defaultdict(OrderedDict)
        self.min_freq = 0

    def _touch(self, key: int) -> None:
        f = self.kf[key]
        del self.buckets[f][key]
        if not self.buckets[f]:
            del self.buckets[f]
            if self.min_freq == f:
                self.min_freq = f + 1
        self.kf[key] = f + 1
        self.buckets[f + 1][key] = None

    def get(self, key: int) -> int:
        if key not in self.kv:
            return -1
        self._touch(key)
        return self.kv[key]

    def put(self, key: int, value: int) -> None:
        if self.cap == 0:
            return
        if key in self.kv:
            self.kv[key] = value
            self._touch(key)
            return
        if len(self.kv) == self.cap:
            evict, _ = self.buckets[self.min_freq].popitem(last=False)
            if not self.buckets[self.min_freq]:
                del self.buckets[self.min_freq]
            del self.kv[evict], self.kf[evict]
        self.kv[key], self.kf[key] = value, 1
        self.buckets[1][key] = None
        self.min_freq = 1


# ---------- LC 208 / 212 Trie ----------

class TrieNode:
    __slots__ = ("children", "end")

    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.end = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.end = True

    def _walk(self, prefix: str) -> Optional[TrieNode]:
        node = self.root
        for ch in prefix:
            node = node.children.get(ch)
            if node is None:
                return None
        return node

    def search(self, word: str) -> bool:
        node = self._walk(word)
        return node is not None and node.end

    def starts_with(self, prefix: str) -> bool:
        return self._walk(prefix) is not None


def find_words(board: List[List[str]], words: List[str]) -> List[str]:
    """LC 212. 所有单词建 Trie，网格 DFS 沿 Trie 走；命中后置 end=False 去重，叶子回收剪枝。"""
    root = TrieNode()
    for w in words:
        node = root
        for ch in w:
            node = node.children.setdefault(ch, TrieNode())
        node.end = True
    m, n = len(board), len(board[0])
    out: List[str] = []

    def dfs(i, j, parent, path):
        ch = board[i][j]
        node = parent.children.get(ch)
        if node is None:
            return
        path += ch
        if node.end:
            out.append(path)
            node.end = False
        board[i][j] = "#"
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            x, y = i + di, j + dj
            if 0 <= x < m and 0 <= y < n and board[x][y] != "#":
                dfs(x, y, node, path)
        board[i][j] = ch
        if not node.children and not node.end:      # 叶子已用完，从 Trie 里摘掉
            del parent.children[ch]

    for i in range(m):
        for j in range(n):
            dfs(i, j, root, "")
    return out


# ---------- LC 380 O(1) 插入删除随机 ----------

class RandomizedSet:
    """数组 + 值到下标的哈希：删除时把末尾换到被删位置。"""

    def __init__(self):
        self.vals: List[int] = []
        self.pos: Dict[int, int] = {}

    def insert(self, val: int) -> bool:
        if val in self.pos:
            return False
        self.pos[val] = len(self.vals)
        self.vals.append(val)
        return True

    def remove(self, val: int) -> bool:
        if val not in self.pos:
            return False
        i, last = self.pos[val], self.vals[-1]
        self.vals[i], self.pos[last] = last, i
        self.vals.pop()
        del self.pos[val]
        return True

    def get_random(self) -> int:
        return random.choice(self.vals)


# ---------- LC 355 设计推特 ----------

class Twitter:
    """每人一条按时间倒序的推文列表；getNewsFeed 用堆做 k 路归并取 10 条。"""

    def __init__(self):
        self.time = 0
        self.tweets: Dict[int, List[tuple]] = defaultdict(list)     # user -> [(-time, tweetId)]
        self.follows: Dict[int, set] = defaultdict(set)

    def post_tweet(self, user: int, tweet: int) -> None:
        self.tweets[user].append((-self.time, tweet))
        self.time += 1

    def get_news_feed(self, user: int) -> List[int]:
        heap = []
        for u in self.follows[user] | {user}:
            if self.tweets[u]:
                t, tid = self.tweets[u][-1]
                heap.append((t, tid, u, len(self.tweets[u]) - 1))
        heapq.heapify(heap)
        out: List[int] = []
        while heap and len(out) < 10:
            t, tid, u, idx = heapq.heappop(heap)
            out.append(tid)
            if idx > 0:
                nt, ntid = self.tweets[u][idx - 1]
                heapq.heappush(heap, (nt, ntid, u, idx - 1))
        return out

    def follow(self, a: int, b: int) -> None:
        if a != b:
            self.follows[a].add(b)

    def unfollow(self, a: int, b: int) -> None:
        self.follows[a].discard(b)


# ---------- LC 307 树状数组 ----------

class NumArray:
    """Fenwick tree：单点更新与前缀和都 O(log n)。"""

    def __init__(self, nums: List[int]):
        self.n = len(nums)
        self.nums = nums[:]
        self.tree = [0] * (self.n + 1)
        for i, x in enumerate(nums):
            self._add(i + 1, x)

    def _add(self, i: int, delta: int) -> None:
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def _prefix(self, i: int) -> int:
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s

    def update(self, index: int, val: int) -> None:
        self._add(index + 1, val - self.nums[index])
        self.nums[index] = val

    def sum_range(self, left: int, right: int) -> int:
        return self._prefix(right + 1) - self._prefix(left)


# ---------- LC 232 两个栈实现队列 ----------

class MyQueue:
    """入栈 in、出栈 out；out 空时把 in 全倒过去。均摊 O(1)。"""

    def __init__(self):
        self.inbox: List[int] = []
        self.outbox: List[int] = []

    def push(self, x: int) -> None:
        self.inbox.append(x)

    def _shift(self) -> None:
        if not self.outbox:
            while self.inbox:
                self.outbox.append(self.inbox.pop())

    def pop(self) -> int:
        self._shift()
        return self.outbox.pop()

    def peek(self) -> int:
        self._shift()
        return self.outbox[-1]

    def empty(self) -> bool:
        return not self.inbox and not self.outbox


# ---------- LC 981 基于时间的键值存储 ----------

class TimeMap:
    """每个 key 一个按时间递增的列表，get 用 bisect_right - 1。"""

    def __init__(self):
        self.store: Dict[str, List[tuple]] = defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.store[key].append((timestamp, value))

    def get(self, key: str, timestamp: int) -> str:
        arr = self.store.get(key, [])
        i = bisect_right(arr, (timestamp, chr(0x10FFFF)))
        return arr[i - 1][1] if i else ""


class SortedList:
    """有序表的极简替身（bisect + insort，插入 O(n)）——面试里说明 Java 用 TreeMap、Python 用 sortedcontainers。"""

    def __init__(self):
        self.a: List[int] = []

    def add(self, x: int) -> None:
        insort(self.a, x)

    def remove(self, x: int) -> None:
        i = bisect_left(self.a, x)
        del self.a[i]

    def rank(self, x: int) -> int:
        return bisect_left(self.a, x)


class Tests(unittest.TestCase):
    def test_lru(self):
        for cls in (LRUCache, LRUCacheOrdered):
            c = cls(2)
            c.put(1, 1); c.put(2, 2)
            self.assertEqual(c.get(1), 1)
            c.put(3, 3)
            self.assertEqual(c.get(2), -1)
            c.put(4, 4)
            self.assertEqual(c.get(1), -1)
            self.assertEqual(c.get(3), 3)
            self.assertEqual(c.get(4), 4)

    def test_lfu(self):
        c = LFUCache(2)
        c.put(1, 1); c.put(2, 2)
        self.assertEqual(c.get(1), 1)
        c.put(3, 3)                       # 淘汰 2（频次 1，最少）
        self.assertEqual(c.get(2), -1)
        self.assertEqual(c.get(3), 3)
        c.put(4, 4)                       # 1 与 3 频次都是 2，淘汰更久没用的 1
        self.assertEqual(c.get(1), -1)
        self.assertEqual(c.get(3), 3)
        self.assertEqual(c.get(4), 4)

    def test_trie(self):
        t = Trie()
        t.insert("apple")
        self.assertTrue(t.search("apple"))
        self.assertFalse(t.search("app"))
        self.assertTrue(t.starts_with("app"))
        board = [list("oaan"), list("etae"), list("ihkr"), list("iflv")]
        self.assertEqual(sorted(find_words(board, ["oath", "pea", "eat", "rain"])), ["eat", "oath"])
        self.assertEqual(find_words([list("ab"), list("cd")], ["abcb"]), [])

    def test_randomized_set(self):
        s = RandomizedSet()
        self.assertTrue(s.insert(1)); self.assertFalse(s.remove(2)); self.assertTrue(s.insert(2))
        self.assertIn(s.get_random(), (1, 2))
        self.assertTrue(s.remove(1)); self.assertFalse(s.insert(2))
        self.assertEqual(s.get_random(), 2)

    def test_twitter(self):
        tw = Twitter()
        tw.post_tweet(1, 5)
        self.assertEqual(tw.get_news_feed(1), [5])
        tw.follow(1, 2); tw.post_tweet(2, 6)
        self.assertEqual(tw.get_news_feed(1), [6, 5])
        tw.unfollow(1, 2)
        self.assertEqual(tw.get_news_feed(1), [5])

    def test_fenwick(self):
        na = NumArray([1, 3, 5])
        self.assertEqual(na.sum_range(0, 2), 9)
        na.update(1, 2)
        self.assertEqual(na.sum_range(0, 2), 8)
        self.assertEqual(na.sum_range(1, 1), 2)

    def test_queue_timemap(self):
        q = MyQueue()
        q.push(1); q.push(2)
        self.assertEqual(q.peek(), 1)
        self.assertEqual(q.pop(), 1)
        self.assertFalse(q.empty())
        tm = TimeMap()
        tm.set("foo", "bar", 1)
        self.assertEqual(tm.get("foo", 1), "bar")
        self.assertEqual(tm.get("foo", 3), "bar")
        tm.set("foo", "bar2", 4)
        self.assertEqual(tm.get("foo", 4), "bar2")
        self.assertEqual(tm.get("foo", 5), "bar2")
        self.assertEqual(tm.get("foo", 0), "")
        sl = SortedList()
        for x in (5, 1, 3):
            sl.add(x)
        self.assertEqual(sl.a, [1, 3, 5]); sl.remove(3); self.assertEqual(sl.rank(5), 1)


if __name__ == "__main__":
    unittest.main(verbosity=1)
