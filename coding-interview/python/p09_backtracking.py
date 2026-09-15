"""面试手撕代码（09）：回溯 —— 配套解法与测试。

https://arganzheng.life/coding-interview-backtracking.html

统一模板：path 记录当前选择；for 候选：做选择 → 递归 → 撤销。
"""
from __future__ import annotations

import unittest
from typing import List


def permute(nums: List[int]) -> List[List[int]]:
    """LC 46. used 数组标记；O(n · n!)。"""
    out, path, used = [], [], [False] * len(nums)

    def bt():
        if len(path) == len(nums):
            out.append(path[:])              # 必须拷贝：path 之后会被改
            return
        for i, x in enumerate(nums):
            if used[i]:
                continue
            used[i] = True
            path.append(x)
            bt()
            path.pop()
            used[i] = False

    bt()
    return out


def permute_unique(nums: List[int]) -> List[List[int]]:
    """LC 47. 排序后：同层里"前一个相同且未使用"则跳过。"""
    nums.sort()
    out, path, used = [], [], [False] * len(nums)

    def bt():
        if len(path) == len(nums):
            out.append(path[:])
            return
        for i, x in enumerate(nums):
            if used[i] or (i > 0 and nums[i] == nums[i - 1] and not used[i - 1]):
                continue
            used[i] = True
            path.append(x)
            bt()
            path.pop()
            used[i] = False

    bt()
    return out


def subsets(nums: List[int]) -> List[List[int]]:
    """LC 78. 每个节点都是一个答案；用 start 保证不回头。O(n · 2^n)。"""
    out, path = [], []

    def bt(start):
        out.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            bt(i + 1)
            path.pop()

    bt(0)
    return out


def subsets_with_dup(nums: List[int]) -> List[List[int]]:
    """LC 90. 排序 + 同层去重（i > start 且与前一个相同则跳过）。"""
    nums.sort()
    out, path = [], []

    def bt(start):
        out.append(path[:])
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue
            path.append(nums[i])
            bt(i + 1)
            path.pop()

    bt(0)
    return out


def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    """LC 39. 可重复选：递归传 i 不是 i+1；排序后 candidates[i] > remain 就 break 剪枝。"""
    candidates.sort()
    out, path = [], []

    def bt(start, remain):
        if remain == 0:
            out.append(path[:])
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remain:
                break
            path.append(candidates[i])
            bt(i, remain - candidates[i])
            path.pop()

    bt(0, target)
    return out


def combination_sum2(candidates: List[int], target: int) -> List[List[int]]:
    """LC 40. 每个数只能用一次且有重复：i+1 + 同层去重。"""
    candidates.sort()
    out, path = [], []

    def bt(start, remain):
        if remain == 0:
            out.append(path[:])
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remain:
                break
            if i > start and candidates[i] == candidates[i - 1]:
                continue
            path.append(candidates[i])
            bt(i + 1, remain - candidates[i])
            path.pop()

    bt(0, target)
    return out


def generate_parenthesis(n: int) -> List[str]:
    """LC 22. 约束式剪枝：左括号数 < n 可放左；右 < 左 可放右。"""
    out: List[str] = []

    def bt(s, open_, close):
        if len(s) == 2 * n:
            out.append(s)
            return
        if open_ < n:
            bt(s + "(", open_ + 1, close)
        if close < open_:
            bt(s + ")", open_, close + 1)

    bt("", 0, 0)
    return out


def partition_palindrome(s: str) -> List[List[str]]:
    """LC 131. 切分型回溯 + 预处理回文表 is_pal[i][j]。"""
    n = len(s)
    is_pal = [[False] * n for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for j in range(i, n):
            is_pal[i][j] = s[i] == s[j] and (j - i < 2 or is_pal[i + 1][j - 1])
    out, path = [], []

    def bt(start):
        if start == n:
            out.append(path[:])
            return
        for end in range(start, n):
            if is_pal[start][end]:
                path.append(s[start:end + 1])
                bt(end + 1)
                path.pop()

    bt(0)
    return out


def exist(board: List[List[str]], word: str) -> bool:
    """LC 79. 网格 DFS，原地标记 '#' 再恢复。O(mn · 3^L)。"""
    m, n = len(board), len(board[0])

    def dfs(i, j, k):
        if k == len(word):
            return True
        if not (0 <= i < m and 0 <= j < n) or board[i][j] != word[k]:
            return False
        tmp, board[i][j] = board[i][j], "#"
        found = (dfs(i + 1, j, k + 1) or dfs(i - 1, j, k + 1) or
                 dfs(i, j + 1, k + 1) or dfs(i, j - 1, k + 1))
        board[i][j] = tmp
        return found

    return any(dfs(i, j, 0) for i in range(m) for j in range(n))


def solve_n_queens(n: int) -> List[List[str]]:
    """LC 51. 逐行放置，用三个集合记录列 / 主对角线 / 副对角线。"""
    out: List[List[str]] = []
    cols, diag1, diag2 = set(), set(), set()
    queens = [-1] * n

    def bt(r):
        if r == n:
            out.append(["." * c + "Q" + "." * (n - c - 1) for c in queens])
            return
        for c in range(n):
            if c in cols or (r - c) in diag1 or (r + c) in diag2:
                continue
            cols.add(c); diag1.add(r - c); diag2.add(r + c)
            queens[r] = c
            bt(r + 1)
            cols.remove(c); diag1.remove(r - c); diag2.remove(r + c)

    bt(0)
    return out


def letter_combinations(digits: str) -> List[str]:
    """LC 17. 多阶段笛卡尔积。"""
    if not digits:
        return []
    keys = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
    out: List[str] = []

    def bt(i, s):
        if i == len(digits):
            out.append(s)
            return
        for ch in keys[digits[i]]:
            bt(i + 1, s + ch)

    bt(0, "")
    return out


class Tests(unittest.TestCase):
    def test_permutations(self):
        self.assertEqual(sorted(permute([1, 2, 3])), [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]])
        self.assertEqual(permute_unique([1, 1, 2]), [[1, 1, 2], [1, 2, 1], [2, 1, 1]])
        self.assertEqual(len(permute_unique([1, 2, 2, 2])), 4)

    def test_subsets(self):
        self.assertEqual(len(subsets([1, 2, 3])), 8)
        self.assertEqual(subsets_with_dup([1, 2, 2]), [[], [1], [1, 2], [1, 2, 2], [2], [2, 2]])

    def test_combination_sum(self):
        self.assertEqual(combination_sum([2, 3, 6, 7], 7), [[2, 2, 3], [7]])
        self.assertEqual(combination_sum([2], 1), [])
        self.assertEqual(combination_sum2([10, 1, 2, 7, 6, 1, 5], 8), [[1, 1, 6], [1, 2, 5], [1, 7], [2, 6]])

    def test_parenthesis(self):
        self.assertEqual(sorted(generate_parenthesis(3)), ["((()))", "(()())", "(())()", "()(())", "()()()"])

    def test_partition(self):
        self.assertEqual(partition_palindrome("aab"), [["a", "a", "b"], ["aa", "b"]])

    def test_word_search(self):
        b = [list("ABCE"), list("SFCS"), list("ADEE")]
        self.assertTrue(exist(b, "ABCCED"))
        self.assertTrue(exist(b, "SEE"))
        self.assertFalse(exist(b, "ABCB"))

    def test_n_queens(self):
        self.assertEqual(len(solve_n_queens(4)), 2)
        self.assertEqual(len(solve_n_queens(8)), 92)
        self.assertIn([".Q..", "...Q", "Q...", "..Q."], solve_n_queens(4))

    def test_letter_combinations(self):
        self.assertEqual(letter_combinations("23"), ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"])
        self.assertEqual(letter_combinations(""), [])


if __name__ == "__main__":
    unittest.main(verbosity=1)
