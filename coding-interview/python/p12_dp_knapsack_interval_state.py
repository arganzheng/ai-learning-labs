"""面试手撕代码（12）：动态规划（二）背包、区间、状态机、树形 —— 配套解法与测试。

https://arganzheng.life/coding-interview-dynamic-programming-knapsack-interval-state-machine.html
"""
from __future__ import annotations

import unittest
from functools import lru_cache
from typing import List, Optional


# ---------- 背包 ----------

def can_partition(nums: List[int]) -> bool:
    """LC 416. 0/1 背包可行性：容量 sum/2，倒序枚举容量。O(n · S) / O(S)。"""
    total = sum(nums)
    if total % 2:
        return False
    target = total // 2
    f = [True] + [False] * target
    for x in nums:
        for c in range(target, x - 1, -1):        # 倒序：每个物品只用一次
            f[c] = f[c] or f[c - x]
    return f[target]


def find_target_sum_ways(nums: List[int], target: int) -> int:
    """LC 494. 正号子集和 P：P - (S-P) = target → P = (S+target)/2，0/1 背包计数。"""
    s = sum(nums)
    if (s + target) % 2 or abs(target) > s:
        return 0
    cap = (s + target) // 2
    f = [1] + [0] * cap
    for x in nums:
        for c in range(cap, x - 1, -1):
            f[c] += f[c - x]
    return f[cap]


def change(amount: int, coins: List[int]) -> int:
    """LC 518. 完全背包计数：正序枚举容量（可重复选）。"""
    f = [1] + [0] * amount
    for c in coins:
        for a in range(c, amount + 1):            # 正序：同一硬币可反复用
            f[a] += f[a - c]
    return f[amount]


def num_squares(n: int) -> int:
    """LC 279. 完全背包最少物品数。"""
    f = [0] + [n] * n
    for i in range(1, n + 1):
        j = 1
        while j * j <= i:
            f[i] = min(f[i], f[i - j * j] + 1)
            j += 1
    return f[n]


# ---------- 区间 DP ----------

def max_coins(nums: List[int]) -> int:
    """LC 312. 戳气球：反着想成"最后戳哪个"，f(i,j) 开区间 (i,j) 的最大得分。O(n^3)。"""
    a = [1] + nums + [1]
    n = len(a)
    f = [[0] * n for _ in range(n)]
    for length in range(2, n):                    # j - i >= 2 才有气球可戳
        for i in range(n - length):
            j = i + length
            for k in range(i + 1, j):             # k 是 (i,j) 中最后戳破的
                f[i][j] = max(f[i][j], f[i][k] + f[k][j] + a[i] * a[k] * a[j])
    return f[0][n - 1]


def longest_palindrome_subseq(s: str) -> int:
    """LC 516. f(i,j) = s[i]==s[j] ? f(i+1,j-1)+2 : max(f(i+1,j), f(i,j-1))。O(n^2)。"""
    n = len(s)
    f = [[0] * n for _ in range(n)]
    for i in range(n - 1, -1, -1):
        f[i][i] = 1
        for j in range(i + 1, n):
            if s[i] == s[j]:
                f[i][j] = f[i + 1][j - 1] + 2
            else:
                f[i][j] = max(f[i + 1][j], f[i][j - 1])
    return f[0][n - 1]


def min_cost_merge_stones(stones: List[int]) -> int:
    """经典石子合并（相邻两堆）：f(i,j) = min(f(i,k)+f(k+1,j)) + sum(i..j)。O(n^3)。"""
    n = len(stones)
    pre = [0]
    for x in stones:
        pre.append(pre[-1] + x)
    f = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            f[i][j] = min(f[i][k] + f[k + 1][j] for k in range(i, j)) + pre[j + 1] - pre[i]
    return f[0][n - 1]


# ---------- 状态机 DP：买卖股票 ----------

def max_profit_one(prices: List[int]) -> int:
    """LC 121. 一次交易：维护最低买入价。"""
    lo, best = float("inf"), 0
    for p in prices:
        lo = min(lo, p)
        best = max(best, p - lo)
    return best


def max_profit_k(k: int, prices: List[int]) -> int:
    """LC 188（含 123 k=2）。hold[j] / free[j]：第 j 笔交易持有 / 空仓的最大收益。O(nk)。"""
    if not prices:
        return 0
    if k >= len(prices) // 2:                     # 交易次数不受限，退化为 LC 122
        return sum(max(0, b - a) for a, b in zip(prices, prices[1:]))
    hold = [float("-inf")] * (k + 1)
    free = [0] * (k + 1)
    for p in prices:
        for j in range(k, 0, -1):
            free[j] = max(free[j], hold[j] + p)
            hold[j] = max(hold[j], free[j - 1] - p)
    return free[k]


def max_profit_cooldown(prices: List[int]) -> int:
    """LC 309. 三状态：hold / sold（今天卖出，明天冻结） / rest。"""
    hold, sold, rest = float("-inf"), 0, 0
    for p in prices:
        hold, sold, rest = max(hold, rest - p), hold + p, max(rest, sold)
    return max(sold, rest)


def max_profit_fee(prices: List[int], fee: int) -> int:
    """LC 714. 两状态，卖出时扣手续费。"""
    hold, free = float("-inf"), 0
    for p in prices:
        hold, free = max(hold, free - p), max(free, hold + p - fee)
    return free


# ---------- 树形 DP ----------

class TreeNode:
    def __init__(self, val: int = 0, left: "Optional[TreeNode]" = None, right: "Optional[TreeNode]" = None):
        self.val, self.left, self.right = val, left, right


def rob_tree(root: Optional[TreeNode]) -> int:
    """LC 337. 后序返回 (偷当前, 不偷当前)。O(n)。"""
    def dfs(node):
        if not node:
            return 0, 0
        l_take, l_skip = dfs(node.left)
        r_take, r_skip = dfs(node.right)
        take = node.val + l_skip + r_skip
        skip = max(l_take, l_skip) + max(r_take, r_skip)
        return take, skip
    return max(dfs(root))


# ---------- 字符串匹配 DP ----------

def is_match_regex(s: str, p: str) -> bool:
    """LC 10. '.' 与 'x*'：f(i,j) 表示 s[i:] 与 p[j:] 匹配（记忆化）。"""
    @lru_cache(maxsize=None)
    def f(i, j):
        if j == len(p):
            return i == len(s)
        first = i < len(s) and p[j] in (s[i], ".")
        if j + 1 < len(p) and p[j + 1] == "*":
            return f(i, j + 2) or (first and f(i + 1, j))   # 跳过 x*，或吃掉一个字符继续用 x*
        return first and f(i + 1, j + 1)
    return f(0, 0)


def is_match_wildcard(s: str, p: str) -> bool:
    """LC 44. '?' 与 '*'：f(i,j) = s[:i] 与 p[:j] 匹配。O(mn)。"""
    m, n = len(s), len(p)
    f = [[False] * (n + 1) for _ in range(m + 1)]
    f[0][0] = True
    for j in range(1, n + 1):
        f[0][j] = f[0][j - 1] and p[j - 1] == "*"
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == "*":
                f[i][j] = f[i][j - 1] or f[i - 1][j]        # * 匹配空 / 再吃一个字符
            else:
                f[i][j] = f[i - 1][j - 1] and p[j - 1] in (s[i - 1], "?")
    return f[m][n]


# ---------- 状压 DP ----------

def shortest_path_visiting_all(n: int, adj: List[List[int]]) -> int:
    """LC 847. 状态 (node, visited_mask) 上的 BFS。O(n · 2^n)。"""
    from collections import deque
    full = (1 << n) - 1
    q = deque((i, 1 << i, 0) for i in range(n))
    seen = {(i, 1 << i) for i in range(n)}
    while q:
        u, mask, d = q.popleft()
        if mask == full:
            return d
        for v in adj[u]:
            nm = mask | (1 << v)
            if (v, nm) not in seen:
                seen.add((v, nm))
                q.append((v, nm, d + 1))
    return 0


class Tests(unittest.TestCase):
    def test_knapsack(self):
        self.assertTrue(can_partition([1, 5, 11, 5]))
        self.assertFalse(can_partition([1, 2, 3, 5]))
        self.assertEqual(find_target_sum_ways([1, 1, 1, 1, 1], 3), 5)
        self.assertEqual(find_target_sum_ways([1], 2), 0)
        self.assertEqual(change(5, [1, 2, 5]), 4)
        self.assertEqual(change(3, [2]), 0)
        self.assertEqual(num_squares(12), 3)
        self.assertEqual(num_squares(13), 2)

    def test_interval(self):
        self.assertEqual(max_coins([3, 1, 5, 8]), 167)
        self.assertEqual(max_coins([1, 5]), 10)
        self.assertEqual(longest_palindrome_subseq("bbbab"), 4)
        self.assertEqual(longest_palindrome_subseq("cbbd"), 2)
        self.assertEqual(min_cost_merge_stones([4, 1, 1, 4]), 18)

    def test_stocks(self):
        self.assertEqual(max_profit_one([7, 1, 5, 3, 6, 4]), 5)
        self.assertEqual(max_profit_one([7, 6, 4, 3, 1]), 0)
        self.assertEqual(max_profit_k(2, [3, 3, 5, 0, 0, 3, 1, 4]), 6)       # LC 123
        self.assertEqual(max_profit_k(2, [1, 2, 3, 4, 5]), 4)
        self.assertEqual(max_profit_k(2, [2, 4, 1]), 2)                      # LC 188
        self.assertEqual(max_profit_k(2, [3, 2, 6, 5, 0, 3]), 7)
        self.assertEqual(max_profit_cooldown([1, 2, 3, 0, 2]), 3)
        self.assertEqual(max_profit_fee([1, 3, 2, 8, 4, 9], 2), 8)

    def test_tree(self):
        t = TreeNode(3, TreeNode(2, None, TreeNode(3)), TreeNode(3, None, TreeNode(1)))
        self.assertEqual(rob_tree(t), 7)
        t = TreeNode(3, TreeNode(4, TreeNode(1), TreeNode(3)), TreeNode(5, None, TreeNode(1)))
        self.assertEqual(rob_tree(t), 9)

    def test_matching(self):
        self.assertFalse(is_match_regex("aa", "a"))
        self.assertTrue(is_match_regex("aa", "a*"))
        self.assertTrue(is_match_regex("ab", ".*"))
        self.assertFalse(is_match_regex("mississippi", "mis*is*p*."))
        self.assertTrue(is_match_regex("aab", "c*a*b"))
        self.assertFalse(is_match_wildcard("aa", "a"))
        self.assertTrue(is_match_wildcard("aa", "*"))
        self.assertFalse(is_match_wildcard("cb", "?a"))
        self.assertTrue(is_match_wildcard("adceb", "*a*b"))
        self.assertFalse(is_match_wildcard("acdcb", "a*c?b"))

    def test_bitmask(self):
        self.assertEqual(shortest_path_visiting_all(4, [[1, 2, 3], [0], [0], [0]]), 4)
        self.assertEqual(shortest_path_visiting_all(5, [[1], [0, 2, 4], [1, 3, 4], [2], [1, 2]]), 4)


if __name__ == "__main__":
    unittest.main(verbosity=1)
