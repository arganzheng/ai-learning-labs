"""面试手撕代码（02）：双指针与滑动窗口 —— 配套解法与测试。

https://arganzheng.life/coding-interview-two-pointers-and-sliding-window.html
"""
from __future__ import annotations

import unittest
from collections import Counter, defaultdict
from typing import List


def length_of_longest_substring(s: str) -> int:
    """LC 3. 变长窗口：右扩、违规则左收。O(n) / O(字符集)。"""
    last: dict[str, int] = {}
    left = best = 0
    for right, ch in enumerate(s):
        if ch in last and last[ch] >= left:
            left = last[ch] + 1            # 直接跳到重复字符的下一位
        last[ch] = right
        best = max(best, right - left + 1)
    return best


def min_window(s: str, t: str) -> str:
    """LC 76. 最小覆盖子串：need 计数 + 已满足字符数 formed。O(|s| + |t|) / O(字符集)。"""
    need = Counter(t)
    missing = len(t)                       # 还差多少个字符（含重复）
    left = 0
    best = (0, float("inf"))
    for right, ch in enumerate(s):
        if need[ch] > 0:
            missing -= 1
        need[ch] -= 1
        if missing == 0:                   # 窗口已覆盖 t，尝试收左边
            while need[s[left]] < 0:       # 左端字符多余
                need[s[left]] += 1
                left += 1
            if right - left < best[1] - best[0]:
                best = (left, right)
            need[s[left]] += 1             # 主动破坏窗口，继续找更短的
            missing += 1
            left += 1
    return "" if best[1] == float("inf") else s[best[0]: best[1] + 1]


def character_replacement(s: str, k: int) -> int:
    """LC 424. 窗口长 - 最高频 <= k 即合法；窗口只增不缩（答案单调）。O(n) / O(26)。"""
    count = defaultdict(int)
    left = max_freq = 0
    for right, ch in enumerate(s):
        count[ch] += 1
        max_freq = max(max_freq, count[ch])   # 历史最大即可：窗口不会缩短，答案不会变差
        if right - left + 1 - max_freq > k:
            count[s[left]] -= 1
            left += 1
    return len(s) - left


def min_sub_array_len(target: int, nums: List[int]) -> int:
    """LC 209. 正数数组：和 >= target 的最短窗口。O(n) / O(1)。"""
    left = total = 0
    best = float("inf")
    for right, x in enumerate(nums):
        total += x
        while total >= target:
            best = min(best, right - left + 1)
            total -= nums[left]
            left += 1
    return 0 if best == float("inf") else best


def three_sum(nums: List[int]) -> List[List[int]]:
    """LC 15. 排序 + 固定一个 + 对撞双指针，三处去重。O(n^2) / O(log n) 排序栈。"""
    nums.sort()
    n = len(nums)
    out: List[List[int]] = []
    for i in range(n - 2):
        if nums[i] > 0:
            break
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        lo, hi = i + 1, n - 1
        while lo < hi:
            s = nums[i] + nums[lo] + nums[hi]
            if s < 0:
                lo += 1
            elif s > 0:
                hi -= 1
            else:
                out.append([nums[i], nums[lo], nums[hi]])
                lo += 1
                hi -= 1
                while lo < hi and nums[lo] == nums[lo - 1]:
                    lo += 1
                while lo < hi and nums[hi] == nums[hi + 1]:
                    hi -= 1
    return out


def max_area(height: List[int]) -> int:
    """LC 11. 对撞双指针：移动矮的那一侧才可能变大。O(n) / O(1)。"""
    lo, hi, best = 0, len(height) - 1, 0
    while lo < hi:
        best = max(best, min(height[lo], height[hi]) * (hi - lo))
        if height[lo] < height[hi]:
            lo += 1
        else:
            hi -= 1
    return best


def trap(height: List[int]) -> int:
    """LC 42. 双指针：矮的一侧的水位由自己这侧的最大值决定。O(n) / O(1)。"""
    lo, hi = 0, len(height) - 1
    left_max = right_max = water = 0
    while lo < hi:
        if height[lo] < height[hi]:
            left_max = max(left_max, height[lo])
            water += left_max - height[lo]
            lo += 1
        else:
            right_max = max(right_max, height[hi])
            water += right_max - height[hi]
            hi -= 1
    return water


def check_inclusion(s1: str, s2: str) -> bool:
    """LC 567. 定长窗口 + 计数差。O(n) / O(26)。"""
    k = len(s1)
    if k > len(s2):
        return False
    need = Counter(s1)
    window = Counter(s2[:k])
    if window == need:
        return True
    for i in range(k, len(s2)):
        window[s2[i]] += 1
        out = s2[i - k]
        window[out] -= 1
        if window[out] == 0:
            del window[out]
        if window == need:
            return True
    return False


def move_zeroes(nums: List[int]) -> None:
    """LC 283. 快慢指针原地分区。O(n) / O(1)。"""
    slow = 0
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1


class Tests(unittest.TestCase):
    def test_longest_substring(self):
        self.assertEqual(length_of_longest_substring("abcabcbb"), 3)
        self.assertEqual(length_of_longest_substring("bbbbb"), 1)
        self.assertEqual(length_of_longest_substring("pwwkew"), 3)
        self.assertEqual(length_of_longest_substring("abba"), 2)
        self.assertEqual(length_of_longest_substring(""), 0)

    def test_min_window(self):
        self.assertEqual(min_window("ADOBECODEBANC", "ABC"), "BANC")
        self.assertEqual(min_window("a", "a"), "a")
        self.assertEqual(min_window("a", "aa"), "")

    def test_character_replacement(self):
        self.assertEqual(character_replacement("ABAB", 2), 4)
        self.assertEqual(character_replacement("AABABBA", 1), 4)

    def test_min_sub_array_len(self):
        self.assertEqual(min_sub_array_len(7, [2, 3, 1, 2, 4, 3]), 2)
        self.assertEqual(min_sub_array_len(11, [1, 1, 1, 1, 1, 1, 1, 1]), 0)

    def test_three_sum(self):
        self.assertEqual(three_sum([-1, 0, 1, 2, -1, -4]), [[-1, -1, 2], [-1, 0, 1]])
        self.assertEqual(three_sum([0, 0, 0, 0]), [[0, 0, 0]])
        self.assertEqual(three_sum([1, 2, 3]), [])

    def test_max_area(self):
        self.assertEqual(max_area([1, 8, 6, 2, 5, 4, 8, 3, 7]), 49)

    def test_trap(self):
        self.assertEqual(trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]), 6)
        self.assertEqual(trap([4, 2, 0, 3, 2, 5]), 9)

    def test_check_inclusion(self):
        self.assertTrue(check_inclusion("ab", "eidbaooo"))
        self.assertFalse(check_inclusion("ab", "eidboaoo"))

    def test_move_zeroes(self):
        a = [0, 1, 0, 3, 12]
        move_zeroes(a)
        self.assertEqual(a, [1, 3, 12, 0, 0])


if __name__ == "__main__":
    unittest.main(verbosity=1)
