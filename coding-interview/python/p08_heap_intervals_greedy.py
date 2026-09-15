"""面试手撕代码（08）：堆、Top-K、区间与贪心 —— 配套解法与测试。

https://arganzheng.life/coding-interview-heap-topk-intervals-greedy.html
"""
from __future__ import annotations

import heapq
import random
import unittest
from collections import Counter
from typing import List


def find_kth_largest_heap(nums: List[int], k: int) -> int:
    """LC 215. 大小为 k 的最小堆：堆顶即第 k 大。O(n log k) / O(k)。"""
    heap: List[int] = []
    for x in nums:
        if len(heap) < k:
            heapq.heappush(heap, x)
        elif x > heap[0]:
            heapq.heapreplace(heap, x)
    return heap[0]


def find_kth_largest_quickselect(nums: List[int], k: int) -> int:
    """LC 215. 快速选择（随机 pivot，三路分区）。期望 O(n) / O(1)。"""
    target = len(nums) - k                      # 第 k 大 = 升序第 target 个（0-based）
    lo, hi = 0, len(nums) - 1
    while True:
        pivot = nums[random.randint(lo, hi)]
        lt, i, gt = lo, lo, hi                  # [lo,lt) < p, [lt,i) == p, (gt,hi] > p
        while i <= gt:
            if nums[i] < pivot:
                nums[lt], nums[i] = nums[i], nums[lt]
                lt += 1
                i += 1
            elif nums[i] > pivot:
                nums[gt], nums[i] = nums[i], nums[gt]
                gt -= 1
            else:
                i += 1
        if target < lt:
            hi = lt - 1
        elif target > gt:
            lo = gt + 1
        else:
            return pivot


def top_k_frequent(nums: List[int], k: int) -> List[int]:
    """LC 347. 桶排序按频次：O(n) / O(n)。（堆解 O(n log k) 见正文）"""
    count = Counter(nums)
    buckets: List[List[int]] = [[] for _ in range(len(nums) + 1)]
    for x, c in count.items():
        buckets[c].append(x)
    out: List[int] = []
    for c in range(len(buckets) - 1, 0, -1):
        for x in buckets[c]:
            out.append(x)
            if len(out) == k:
                return out
    return out


def k_closest(points: List[List[int]], k: int) -> List[List[int]]:
    """LC 973. 大小为 k 的最大堆（Python 取负）。O(n log k)。"""
    heap: List[tuple] = []
    for x, y in points:
        d = -(x * x + y * y)
        if len(heap) < k:
            heapq.heappush(heap, (d, x, y))
        elif d > heap[0][0]:
            heapq.heapreplace(heap, (d, x, y))
    return [[x, y] for _, x, y in heap]


class MedianFinder:
    """LC 295. 两个堆：大顶堆 small 存较小一半、小顶堆 large 存较大一半，|small| >= |large|。"""

    def __init__(self) -> None:
        self.small: List[int] = []   # 取负实现大顶堆
        self.large: List[int] = []

    def add_num(self, num: int) -> None:
        heapq.heappush(self.small, -num)
        heapq.heappush(self.large, -heapq.heappop(self.small))   # 先过一遍 small 保证 small 的最大 <= large 的最小
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def find_median(self) -> float:
        if len(self.small) > len(self.large):
            return float(-self.small[0])
        return (-self.small[0] + self.large[0]) / 2


def merge_intervals(intervals: List[List[int]]) -> List[List[int]]:
    """LC 56. 按起点排序，能接上就扩右端。O(n log n)。"""
    intervals.sort()
    out: List[List[int]] = []
    for s, e in intervals:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def insert_interval(intervals: List[List[int]], new: List[int]) -> List[List[int]]:
    """LC 57. 三段：左边不相交、重叠的合并、右边不相交。O(n)。"""
    out: List[List[int]] = []
    i, n = 0, len(intervals)
    s, e = new
    while i < n and intervals[i][1] < s:
        out.append(intervals[i]); i += 1
    while i < n and intervals[i][0] <= e:
        s, e = min(s, intervals[i][0]), max(e, intervals[i][1]); i += 1
    out.append([s, e])
    out.extend(intervals[i:])
    return out


def erase_overlap_intervals(intervals: List[List[int]]) -> int:
    """LC 435. 按右端点排序贪心：保留结束最早的。O(n log n)。"""
    intervals.sort(key=lambda x: x[1])
    kept, end = 0, float("-inf")
    for s, e in intervals:
        if s >= end:
            kept += 1
            end = e
    return len(intervals) - kept


def find_min_arrow_shots(points: List[List[int]]) -> int:
    """LC 452. 与 435 同一贪心：一箭射穿所有与当前最早右端相交的气球。"""
    points.sort(key=lambda x: x[1])
    arrows, end = 0, float("-inf")
    for s, e in points:
        if s > end:
            arrows += 1
            end = e
    return arrows


def min_meeting_rooms(intervals: List[List[int]]) -> int:
    """LC 253. 按开始排序 + 最小堆存各会议室的结束时间。O(n log n)。"""
    intervals.sort()
    ends: List[int] = []
    for s, e in intervals:
        if ends and ends[0] <= s:
            heapq.heapreplace(ends, e)          # 复用最早空出来的房间
        else:
            heapq.heappush(ends, e)
    return len(ends)


def min_meeting_rooms_sweep(intervals: List[List[int]]) -> int:
    """LC 253 扫描线版：开始 +1、结束 -1，最大重叠数。"""
    events = sorted([(s, 1) for s, _ in intervals] + [(e, -1) for _, e in intervals])   # 同一时刻 -1 排在 +1 前
    cur = best = 0
    for _, d in events:
        cur += d
        best = max(best, cur)
    return best


def least_interval(tasks: List[str], n: int) -> int:
    """LC 621. 数学：以最高频任务为骨架，(max-1)*(n+1)+同频个数，与总数取大。O(n)。"""
    count = Counter(tasks)
    top = max(count.values())
    ties = sum(1 for c in count.values() if c == top)
    return max(len(tasks), (top - 1) * (n + 1) + ties)


def can_jump(nums: List[int]) -> bool:
    """LC 55. 维护能到的最远位置。O(n)。"""
    reach = 0
    for i, x in enumerate(nums):
        if i > reach:
            return False
        reach = max(reach, i + x)
    return True


def jump(nums: List[int]) -> int:
    """LC 45. 按"层"贪心：当前层能到的最远边界，走到边界就跳一步。O(n)。"""
    steps = end = farthest = 0
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == end:
            steps += 1
            end = farthest
    return steps


def can_complete_circuit(gas: List[int], cost: List[int]) -> int:
    """LC 134. 总和 >= 0 必有解；从亏空处的下一站重新开始。O(n)。"""
    total = tank = start = 0
    for i in range(len(gas)):
        diff = gas[i] - cost[i]
        total += diff
        tank += diff
        if tank < 0:
            start = i + 1
            tank = 0
    return start if total >= 0 else -1


def max_profit_unlimited(prices: List[int]) -> int:
    """LC 122. 贪心：所有上坡都吃。"""
    return sum(max(0, b - a) for a, b in zip(prices, prices[1:]))


class Tests(unittest.TestCase):
    def test_kth_largest(self):
        for f in (find_kth_largest_heap, find_kth_largest_quickselect):
            self.assertEqual(f([3, 2, 1, 5, 6, 4], 2), 5)
            self.assertEqual(f([3, 2, 3, 1, 2, 4, 5, 5, 6], 4), 4)
            self.assertEqual(f([1], 1), 1)
        random.seed(0)
        for _ in range(50):
            a = [random.randint(-20, 20) for _ in range(random.randint(1, 30))]
            k = random.randint(1, len(a))
            self.assertEqual(find_kth_largest_quickselect(a[:], k), sorted(a)[-k])

    def test_top_k(self):
        self.assertEqual(sorted(top_k_frequent([1, 1, 1, 2, 2, 3], 2)), [1, 2])
        self.assertEqual(top_k_frequent([1], 1), [1])
        self.assertEqual(sorted(k_closest([[3, 3], [5, -1], [-2, 4]], 2)), [[-2, 4], [3, 3]])

    def test_median(self):
        mf = MedianFinder()
        mf.add_num(1); mf.add_num(2)
        self.assertEqual(mf.find_median(), 1.5)
        mf.add_num(3)
        self.assertEqual(mf.find_median(), 2.0)
        mf.add_num(0); mf.add_num(-1)
        self.assertEqual(mf.find_median(), 1.0)

    def test_intervals(self):
        self.assertEqual(merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]), [[1, 6], [8, 10], [15, 18]])
        self.assertEqual(merge_intervals([[1, 4], [4, 5]]), [[1, 5]])
        self.assertEqual(insert_interval([[1, 3], [6, 9]], [2, 5]), [[1, 5], [6, 9]])
        self.assertEqual(insert_interval([[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]], [4, 8]), [[1, 2], [3, 10], [12, 16]])
        self.assertEqual(erase_overlap_intervals([[1, 2], [2, 3], [3, 4], [1, 3]]), 1)
        self.assertEqual(find_min_arrow_shots([[10, 16], [2, 8], [1, 6], [7, 12]]), 2)

    def test_meeting_rooms(self):
        for f in (min_meeting_rooms, min_meeting_rooms_sweep):
            self.assertEqual(f([[0, 30], [5, 10], [15, 20]]), 2)
            self.assertEqual(f([[7, 10], [2, 4]]), 1)
            self.assertEqual(f([[1, 5], [5, 10]]), 1)

    def test_greedy(self):
        self.assertEqual(least_interval(list("AAABBB"), 2), 8)
        self.assertEqual(least_interval(list("AAABBB"), 0), 6)
        self.assertEqual(least_interval(list("AAAAAABCDEFG"), 2), 16)
        self.assertTrue(can_jump([2, 3, 1, 1, 4]))
        self.assertFalse(can_jump([3, 2, 1, 0, 4]))
        self.assertEqual(jump([2, 3, 1, 1, 4]), 2)
        self.assertEqual(jump([2, 3, 0, 1, 4]), 2)
        self.assertEqual(can_complete_circuit([1, 2, 3, 4, 5], [3, 4, 5, 1, 2]), 3)
        self.assertEqual(can_complete_circuit([2, 3, 4], [3, 4, 3]), -1)
        self.assertEqual(max_profit_unlimited([7, 1, 5, 3, 6, 4]), 7)


if __name__ == "__main__":
    unittest.main(verbosity=1)
