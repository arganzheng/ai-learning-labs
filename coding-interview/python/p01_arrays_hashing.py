"""面试手撕代码（01）：数组、哈希与前缀和 —— 配套解法与测试。

https://arganzheng.life/coding-interview-arrays-hashing-prefix-sum.html

每个函数的 docstring 写题号与复杂度；`python p01_arrays_hashing.py` 跑全部断言。
"""
from __future__ import annotations

import unittest
from collections import defaultdict
from typing import List


def two_sum(nums: List[int], target: int) -> List[int]:
    """LC 1. 一遍哈希：边查边存。O(n) / O(n)。"""
    seen: dict[int, int] = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
    return []


def subarray_sum(nums: List[int], k: int) -> int:
    """LC 560. 前缀和 + 哈希计数：count[pre - k]。O(n) / O(n)。"""
    count = defaultdict(int)
    count[0] = 1                       # 空前缀：让从下标 0 开始的子数组也能被数到
    pre = ans = 0
    for x in nums:
        pre += x
        ans += count[pre - k]          # 先查再存，避免把自己算进去
        count[pre] += 1
    return ans


def longest_consecutive(nums: List[int]) -> int:
    """LC 128. 只从"序列起点"（x-1 不在集合里）向右数。O(n) / O(n)。"""
    s = set(nums)
    best = 0
    for x in s:
        if x - 1 in s:
            continue
        y = x
        while y + 1 in s:
            y += 1
        best = max(best, y - x + 1)
    return best


def first_missing_positive(nums: List[int]) -> int:
    """LC 41. 原地哈希：把值 v 放到下标 v-1。O(n) / O(1)。"""
    n = len(nums)
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            j = nums[i] - 1
            nums[i], nums[j] = nums[j], nums[i]
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    return n + 1


def product_except_self(nums: List[int]) -> List[int]:
    """LC 238. 前缀积正着写、后缀积倒着乘。O(n) / O(1) 额外空间。"""
    n = len(nums)
    out = [1] * n
    for i in range(1, n):
        out[i] = out[i - 1] * nums[i - 1]
    suffix = 1
    for i in range(n - 1, -1, -1):
        out[i] *= suffix
        suffix *= nums[i]
    return out


def corp_flight_bookings(bookings: List[List[int]], n: int) -> List[int]:
    """LC 1109. 差分数组：区间加法 O(1)，最后前缀和还原。O(n + m) / O(n)。"""
    diff = [0] * (n + 1)
    for first, last, seats in bookings:
        diff[first - 1] += seats
        diff[last] -= seats
    out = [0] * n
    run = 0
    for i in range(n):
        run += diff[i]
        out[i] = run
    return out


def single_number(nums: List[int]) -> int:
    """LC 136. 异或：a ^ a = 0。O(n) / O(1)。"""
    x = 0
    for v in nums:
        x ^= v
    return x


def single_number_iii(nums: List[int]) -> List[int]:
    """LC 260. 全体异或得 a^b；取最低位 1 把数分成两组各自异或。O(n) / O(1)。"""
    xor = 0
    for v in nums:
        xor ^= v
    low = xor & -xor
    a = 0
    for v in nums:
        if v & low:
            a ^= v
    return sorted([a, xor ^ a])


def find_duplicate(nums: List[int]) -> int:
    """LC 287. 把下标当指针 —— Floyd 判环。O(n) / O(1)。"""
    slow = fast = nums[0]
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    return slow


class Tests(unittest.TestCase):
    def test_two_sum(self):
        self.assertEqual(two_sum([2, 7, 11, 15], 9), [0, 1])
        self.assertEqual(two_sum([3, 3], 6), [0, 1])

    def test_subarray_sum(self):
        self.assertEqual(subarray_sum([1, 1, 1], 2), 2)
        self.assertEqual(subarray_sum([1, 2, 3], 3), 2)
        self.assertEqual(subarray_sum([1, -1, 0], 0), 3)

    def test_longest_consecutive(self):
        self.assertEqual(longest_consecutive([100, 4, 200, 1, 3, 2]), 4)
        self.assertEqual(longest_consecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]), 9)
        self.assertEqual(longest_consecutive([]), 0)

    def test_first_missing_positive(self):
        self.assertEqual(first_missing_positive([1, 2, 0]), 3)
        self.assertEqual(first_missing_positive([3, 4, -1, 1]), 2)
        self.assertEqual(first_missing_positive([7, 8, 9, 11, 12]), 1)
        self.assertEqual(first_missing_positive([1, 1]), 2)

    def test_product_except_self(self):
        self.assertEqual(product_except_self([1, 2, 3, 4]), [24, 12, 8, 6])
        self.assertEqual(product_except_self([-1, 1, 0, -3, 3]), [0, 0, 9, 0, 0])

    def test_corp_flight_bookings(self):
        self.assertEqual(corp_flight_bookings([[1, 2, 10], [2, 3, 20], [2, 5, 25]], 5), [10, 55, 45, 25, 25])

    def test_bits(self):
        self.assertEqual(single_number([4, 1, 2, 1, 2]), 4)
        self.assertEqual(single_number_iii([1, 2, 1, 3, 2, 5]), [3, 5])

    def test_find_duplicate(self):
        self.assertEqual(find_duplicate([1, 3, 4, 2, 2]), 2)
        self.assertEqual(find_duplicate([3, 1, 3, 4, 2]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=1)
