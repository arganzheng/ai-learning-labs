"""面试手撕代码（11）：动态规划（一）线性与二维 —— 配套解法与测试。

https://arganzheng.life/coding-interview-dynamic-programming-linear-and-grid.html
"""
from __future__ import annotations

import unittest
from bisect import bisect_left
from typing import List


def climb_stairs(n: int) -> int:
    """LC 70. f(i) = f(i-1) + f(i-2)，滚动两个变量。O(n) / O(1)。"""
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


def rob(nums: List[int]) -> int:
    """LC 198. f(i) = max(f(i-1), f(i-2) + nums[i])。O(n) / O(1)。"""
    prev2 = prev1 = 0
    for x in nums:
        prev2, prev1 = prev1, max(prev1, prev2 + x)
    return prev1


def rob_circular(nums: List[int]) -> int:
    """LC 213. 环：要么不偷第一间，要么不偷最后一间，两次线性。"""
    if len(nums) == 1:
        return nums[0]
    return max(rob(nums[1:]), rob(nums[:-1]))


def coin_change(coins: List[int], amount: int) -> int:
    """LC 322. 完全背包求最少硬币：f(a) = min(f(a-c)) + 1。O(amount · coins)。"""
    INF = amount + 1
    f = [0] + [INF] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and f[a - c] + 1 < f[a]:
                f[a] = f[a - c] + 1
    return -1 if f[amount] == INF else f[amount]


def length_of_lis(nums: List[int]) -> int:
    """LC 300. O(n^2)：f(i) = 1 + max(f(j)) for j<i, nums[j]<nums[i]。"""
    n = len(nums)
    f = [1] * n
    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i]:
                f[i] = max(f[i], f[j] + 1)
    return max(f, default=0)


def length_of_lis_patience(nums: List[int]) -> int:
    """LC 300. O(n log n)：tails[k] = 长度 k+1 的上升子序列的最小末尾，二分替换。"""
    tails: List[int] = []
    for x in nums:
        i = bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


def max_sub_array(nums: List[int]) -> int:
    """LC 53. Kadane：f(i) = max(nums[i], f(i-1) + nums[i])。"""
    best = cur = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best


def max_product(nums: List[int]) -> int:
    """LC 152. 同时维护最大与最小（负数翻转）。"""
    best = cur_max = cur_min = nums[0]
    for x in nums[1:]:
        cands = (x, cur_max * x, cur_min * x)
        cur_max, cur_min = max(cands), min(cands)
        best = max(best, cur_max)
    return best


def word_break(s: str, word_dict: List[str]) -> bool:
    """LC 139. f(i): s[:i] 可拆。O(n^2 · 比较)。"""
    words = set(word_dict)
    f = [True] + [False] * len(s)
    for i in range(1, len(s) + 1):
        for j in range(i):
            if f[j] and s[j:i] in words:
                f[i] = True
                break
    return f[-1]


def unique_paths(m: int, n: int) -> int:
    """LC 62. 一维滚动：f[j] += f[j-1]。O(mn) / O(n)。"""
    f = [1] * n
    for _ in range(1, m):
        for j in range(1, n):
            f[j] += f[j - 1]
    return f[-1]


def min_path_sum(grid: List[List[int]]) -> int:
    """LC 64. 原地 DP：grid[i][j] += min(上, 左)。"""
    m, n = len(grid), len(grid[0])
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                continue
            up = grid[i - 1][j] if i else float("inf")
            left = grid[i][j - 1] if j else float("inf")
            grid[i][j] += min(up, left)
    return grid[-1][-1]


def maximal_square(matrix: List[List[str]]) -> int:
    """LC 221. f(i,j) = 以 (i,j) 为右下角的最大正方形边长 = 1 + min(上, 左, 左上)。"""
    m, n = len(matrix), len(matrix[0])
    f = [[0] * (n + 1) for _ in range(m + 1)]
    best = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if matrix[i - 1][j - 1] == "1":
                f[i][j] = 1 + min(f[i - 1][j], f[i][j - 1], f[i - 1][j - 1])
                best = max(best, f[i][j])
    return best * best


def longest_common_subsequence(a: str, b: str) -> int:
    """LC 1143. f(i,j) = a[i-1]==b[j-1] ? f(i-1,j-1)+1 : max(f(i-1,j), f(i,j-1))。O(mn)。"""
    m, n = len(a), len(b)
    f = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                f[i][j] = f[i - 1][j - 1] + 1
            else:
                f[i][j] = max(f[i - 1][j], f[i][j - 1])
    return f[m][n]


def min_distance(a: str, b: str) -> int:
    """LC 72. 编辑距离：相等则继承，否则 1 + min(删, 增, 换)。O(mn)。"""
    m, n = len(a), len(b)
    f = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        f[i][0] = i
    for j in range(n + 1):
        f[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                f[i][j] = f[i - 1][j - 1]
            else:
                f[i][j] = 1 + min(f[i - 1][j], f[i][j - 1], f[i - 1][j - 1])
    return f[m][n]


def longest_palindrome_dp(s: str) -> str:
    """LC 5 的区间 DP 版：is_pal[i][j] = s[i]==s[j] and is_pal[i+1][j-1]。O(n^2) / O(n^2)。"""
    n = len(s)
    if n < 2:
        return s
    is_pal = [[False] * n for _ in range(n)]
    start, best = 0, 1
    for i in range(n - 1, -1, -1):
        for j in range(i, n):
            if s[i] == s[j] and (j - i < 2 or is_pal[i + 1][j - 1]):
                is_pal[i][j] = True
                if j - i + 1 > best:
                    start, best = i, j - i + 1
    return s[start:start + best]


def num_decodings(s: str) -> int:
    """LC 91. f(i) = (s[i-1]!='0') f(i-1) + (10<=s[i-2:i]<=26) f(i-2)。"""
    prev2, prev1 = 1, 1 if s[0] != "0" else 0
    for i in range(2, len(s) + 1):
        cur = 0
        if s[i - 1] != "0":
            cur += prev1
        if 10 <= int(s[i - 2:i]) <= 26:
            cur += prev2
        prev2, prev1 = prev1, cur
    return prev1


class Tests(unittest.TestCase):
    def test_1d(self):
        self.assertEqual(climb_stairs(2), 2)
        self.assertEqual(climb_stairs(5), 8)
        self.assertEqual(rob([2, 7, 9, 3, 1]), 12)
        self.assertEqual(rob_circular([2, 3, 2]), 3)
        self.assertEqual(rob_circular([1, 2, 3, 1]), 4)
        self.assertEqual(coin_change([1, 2, 5], 11), 3)
        self.assertEqual(coin_change([2], 3), -1)
        self.assertEqual(coin_change([1], 0), 0)

    def test_lis(self):
        for f in (length_of_lis, length_of_lis_patience):
            self.assertEqual(f([10, 9, 2, 5, 3, 7, 101, 18]), 4)
            self.assertEqual(f([0, 1, 0, 3, 2, 3]), 4)
            self.assertEqual(f([7, 7, 7]), 1)

    def test_subarrays(self):
        self.assertEqual(max_sub_array([-2, 1, -3, 4, -1, 2, 1, -5, 4]), 6)
        self.assertEqual(max_sub_array([-1]), -1)
        self.assertEqual(max_product([2, 3, -2, 4]), 6)
        self.assertEqual(max_product([-2, 0, -1]), 0)
        self.assertEqual(max_product([-2, 3, -4]), 24)

    def test_word_break(self):
        self.assertTrue(word_break("leetcode", ["leet", "code"]))
        self.assertFalse(word_break("catsandog", ["cats", "dog", "sand", "and", "cat"]))

    def test_grid(self):
        self.assertEqual(unique_paths(3, 7), 28)
        self.assertEqual(unique_paths(3, 2), 3)
        self.assertEqual(min_path_sum([[1, 3, 1], [1, 5, 1], [4, 2, 1]]), 7)
        self.assertEqual(maximal_square([list("10100"), list("10111"), list("11111"), list("10010")]), 4)

    def test_two_strings(self):
        self.assertEqual(longest_common_subsequence("abcde", "ace"), 3)
        self.assertEqual(longest_common_subsequence("abc", "def"), 0)
        self.assertEqual(min_distance("horse", "ros"), 3)
        self.assertEqual(min_distance("intention", "execution"), 5)
        self.assertEqual(min_distance("", "abc"), 3)

    def test_misc(self):
        self.assertIn(longest_palindrome_dp("babad"), ("bab", "aba"))
        self.assertEqual(num_decodings("226"), 3)
        self.assertEqual(num_decodings("06"), 0)
        self.assertEqual(num_decodings("10"), 1)


if __name__ == "__main__":
    unittest.main(verbosity=1)
