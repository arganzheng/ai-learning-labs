"""面试手撕代码（07）：二分 —— 配套解法与测试。

https://arganzheng.life/coding-interview-binary-search.html

所有二分都写成同一个模板：在 [lo, hi) 上找第一个使 pred 为真的位置。
"""
from __future__ import annotations

import heapq
import unittest
from typing import Callable, List


def first_true(lo: int, hi: int, pred: Callable[[int], bool]) -> int:
    """在 [lo, hi) 上返回第一个 pred(i) 为真的 i；全假返回 hi。要求 pred 单调（假…假真…真）。"""
    while lo < hi:
        mid = (lo + hi) // 2
        if pred(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def lower_bound(a: List[int], x: int) -> int:
    """第一个 >= x 的下标（等价 bisect_left）。"""
    return first_true(0, len(a), lambda i: a[i] >= x)


def upper_bound(a: List[int], x: int) -> int:
    """第一个 > x 的下标（等价 bisect_right）。"""
    return first_true(0, len(a), lambda i: a[i] > x)


def search_insert(nums: List[int], target: int) -> int:
    """LC 35."""
    return lower_bound(nums, target)


def search_range(nums: List[int], target: int) -> List[int]:
    """LC 34. 两次 lower_bound。O(log n)。"""
    lo = lower_bound(nums, target)
    if lo == len(nums) or nums[lo] != target:
        return [-1, -1]
    return [lo, upper_bound(nums, target) - 1]


def search_rotated(nums: List[int], target: int) -> int:
    """LC 33. 每次判断 mid 落在哪一段有序区间，再决定 target 在不在那一段。O(log n)。"""
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        if nums[lo] <= nums[mid]:                       # 左半有序
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:                                           # 右半有序
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1


def search_rotated_ii(nums: List[int], target: int) -> bool:
    """LC 81. 有重复：nums[lo] == nums[mid] == nums[hi] 时无法判断，收缩一步。最坏 O(n)。"""
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return True
        if nums[lo] == nums[mid] == nums[hi]:
            lo += 1
            hi -= 1
        elif nums[lo] <= nums[mid]:
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return False


def find_min_rotated(nums: List[int]) -> int:
    """LC 153. 与最右元素比：nums[mid] <= nums[-1] 的第一个位置就是最小值。"""
    i = first_true(0, len(nums) - 1, lambda m: nums[m] <= nums[-1])
    return nums[i]


def find_peak_element(nums: List[int]) -> int:
    """LC 162. 第一个 nums[i] > nums[i+1] 的 i（上坡尽头）。"""
    return first_true(0, len(nums) - 1, lambda m: nums[m] > nums[m + 1])


def my_sqrt(x: int) -> int:
    """LC 69. 最后一个 m*m <= x = (第一个 m*m > x) - 1。"""
    return first_true(0, x + 1, lambda m: m * m > x) - 1


def min_eating_speed(piles: List[int], h: int) -> int:
    """LC 875. 答案二分：速度越大用时越少（单调），找最小可行速度。O(n log max)。"""
    def ok(k):
        return sum((p + k - 1) // k for p in piles) <= h
    return first_true(1, max(piles) + 1, ok)


def ship_within_days(weights: List[int], days: int) -> int:
    """LC 1011. 答案二分：运力下界 max(w)，上界 sum(w)。"""
    def ok(cap):
        used, cur = 1, 0
        for w in weights:
            if cur + w > cap:
                used += 1
                cur = 0
            cur += w
        return used <= days
    return first_true(max(weights), sum(weights) + 1, ok)


def split_array(nums: List[int], k: int) -> int:
    """LC 410. 与 1011 完全同构：最大子数组和 <= cap 时最少能分几段。"""
    def ok(cap):
        parts, cur = 1, 0
        for x in nums:
            if cur + x > cap:
                parts += 1
                cur = 0
            cur += x
        return parts <= k
    return first_true(max(nums), sum(nums) + 1, ok)


def kth_smallest_matrix(matrix: List[List[int]], k: int) -> int:
    """LC 378. 值域二分：数出 <= x 的元素个数（从左下角走阶梯，O(n)）。O(n log 值域)。"""
    n = len(matrix)

    def count_le(x):
        i, j, c = n - 1, 0, 0
        while i >= 0 and j < n:
            if matrix[i][j] <= x:
                c += i + 1
                j += 1
            else:
                i -= 1
        return c

    return first_true(matrix[0][0], matrix[-1][-1] + 1, lambda x: count_le(x) >= k)


def kth_smallest_matrix_heap(matrix: List[List[int]], k: int) -> int:
    """LC 378 的堆解：k 路归并。O(k log n)，与二分对照。"""
    n = len(matrix)
    heap = [(matrix[i][0], i, 0) for i in range(min(n, k))]
    heapq.heapify(heap)
    for _ in range(k - 1):
        _, i, j = heapq.heappop(heap)
        if j + 1 < n:
            heapq.heappush(heap, (matrix[i][j + 1], i, j + 1))
    return heap[0][0]


def find_median_sorted_arrays(a: List[int], b: List[int]) -> float:
    """LC 4. 在短数组上二分切分点 i，使左半 = (m+n+1)//2 个且 max(左) <= min(右)。O(log min(m,n))。"""
    if len(a) > len(b):
        a, b = b, a
    m, n = len(a), len(b)
    half = (m + n + 1) // 2
    lo, hi = 0, m
    while lo <= hi:
        i = (lo + hi) // 2                       # a 的左半取 i 个
        j = half - i                             # b 的左半取 j 个
        a_left = a[i - 1] if i > 0 else float("-inf")
        a_right = a[i] if i < m else float("inf")
        b_left = b[j - 1] if j > 0 else float("-inf")
        b_right = b[j] if j < n else float("inf")
        if a_left <= b_right and b_left <= a_right:
            if (m + n) % 2:
                return float(max(a_left, b_left))
            return (max(a_left, b_left) + min(a_right, b_right)) / 2
        if a_left > b_right:
            hi = i - 1
        else:
            lo = i + 1
    raise ValueError("inputs not sorted")


class Tests(unittest.TestCase):
    def test_bounds(self):
        a = [1, 2, 2, 2, 5, 7]
        self.assertEqual(lower_bound(a, 2), 1)
        self.assertEqual(upper_bound(a, 2), 4)
        self.assertEqual(lower_bound(a, 8), 6)
        self.assertEqual(lower_bound(a, 0), 0)
        self.assertEqual(search_insert([1, 3, 5, 6], 5), 2)
        self.assertEqual(search_insert([1, 3, 5, 6], 2), 1)
        self.assertEqual(search_insert([1, 3, 5, 6], 7), 4)

    def test_search_range(self):
        self.assertEqual(search_range([5, 7, 7, 8, 8, 10], 8), [3, 4])
        self.assertEqual(search_range([5, 7, 7, 8, 8, 10], 6), [-1, -1])
        self.assertEqual(search_range([], 0), [-1, -1])

    def test_rotated(self):
        self.assertEqual(search_rotated([4, 5, 6, 7, 0, 1, 2], 0), 4)
        self.assertEqual(search_rotated([4, 5, 6, 7, 0, 1, 2], 3), -1)
        self.assertEqual(search_rotated([1], 0), -1)
        self.assertTrue(search_rotated_ii([2, 5, 6, 0, 0, 1, 2], 0))
        self.assertFalse(search_rotated_ii([2, 5, 6, 0, 0, 1, 2], 3))
        self.assertTrue(search_rotated_ii([1, 0, 1, 1, 1], 0))
        self.assertEqual(find_min_rotated([3, 4, 5, 1, 2]), 1)
        self.assertEqual(find_min_rotated([4, 5, 6, 7, 0, 1, 2]), 0)
        self.assertEqual(find_min_rotated([11, 13, 15, 17]), 11)

    def test_peak_sqrt(self):
        self.assertEqual(find_peak_element([1, 2, 3, 1]), 2)
        self.assertIn(find_peak_element([1, 2, 1, 3, 5, 6, 4]), (1, 5))
        self.assertEqual(my_sqrt(8), 2)
        self.assertEqual(my_sqrt(0), 0)
        self.assertEqual(my_sqrt(2147395599), 46339)

    def test_answer_binary_search(self):
        self.assertEqual(min_eating_speed([3, 6, 7, 11], 8), 4)
        self.assertEqual(min_eating_speed([30, 11, 23, 4, 20], 5), 30)
        self.assertEqual(ship_within_days([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5), 15)
        self.assertEqual(split_array([7, 2, 5, 10, 8], 2), 18)
        self.assertEqual(split_array([1, 2, 3, 4, 5], 2), 9)

    def test_matrix(self):
        m = [[1, 5, 9], [10, 11, 13], [12, 13, 15]]
        self.assertEqual(kth_smallest_matrix(m, 8), 13)
        self.assertEqual(kth_smallest_matrix_heap(m, 8), 13)
        self.assertEqual(kth_smallest_matrix([[-5]], 1), -5)

    def test_median(self):
        self.assertEqual(find_median_sorted_arrays([1, 3], [2]), 2.0)
        self.assertEqual(find_median_sorted_arrays([1, 2], [3, 4]), 2.5)
        self.assertEqual(find_median_sorted_arrays([], [1]), 1.0)
        self.assertEqual(find_median_sorted_arrays([2], []), 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=1)
