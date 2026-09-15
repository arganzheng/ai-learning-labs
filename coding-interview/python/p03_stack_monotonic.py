"""面试手撕代码（03）：栈、单调栈与单调队列 —— 配套解法与测试。

https://arganzheng.life/coding-interview-stack-monotonic-stack-and-queue.html
"""
from __future__ import annotations

import unittest
from collections import deque
from typing import List


def is_valid(s: str) -> bool:
    """LC 20. 左括号入栈，右括号必须匹配栈顶。O(n) / O(n)。"""
    pair = {")": "(", "]": "[", "}": "{"}
    stack: List[str] = []
    for ch in s:
        if ch in pair:
            if not stack or stack.pop() != pair[ch]:
                return False
        else:
            stack.append(ch)
    return not stack


def eval_rpn(tokens: List[str]) -> int:
    """LC 150. 逆波兰：数字入栈，运算符弹两个。除法向零取整。O(n) / O(n)。"""
    stack: List[int] = []
    for t in tokens:
        if t in "+-*/" and len(t) == 1:
            b, a = stack.pop(), stack.pop()
            if t == "+":
                stack.append(a + b)
            elif t == "-":
                stack.append(a - b)
            elif t == "*":
                stack.append(a * b)
            else:
                stack.append(int(a / b))      # Python 的 // 是向下取整，题目要向零
        else:
            stack.append(int(t))
    return stack[0]


def decode_string(s: str) -> str:
    """LC 394. 遇 '[' 把(当前串, 重复数)压栈，遇 ']' 弹出拼接。O(输出长度)。"""
    stack: List[tuple[str, int]] = []
    cur, num = "", 0
    for ch in s:
        if ch.isdigit():
            num = num * 10 + int(ch)
        elif ch == "[":
            stack.append((cur, num))
            cur, num = "", 0
        elif ch == "]":
            prev, k = stack.pop()
            cur = prev + cur * k
        else:
            cur += ch
    return cur


def daily_temperatures(temps: List[int]) -> List[int]:
    """LC 739. 单调递减栈（存下标）：找右侧第一个更大。O(n) / O(n)。"""
    out = [0] * len(temps)
    stack: List[int] = []
    for i, t in enumerate(temps):
        while stack and temps[stack[-1]] < t:
            j = stack.pop()
            out[j] = i - j
        stack.append(i)
    return out


def largest_rectangle_area(heights: List[int]) -> int:
    """LC 84. 单调递增栈 + 哨兵：弹出时左右边界都已知。O(n) / O(n)。"""
    hs = [0] + heights + [0]                # 两端哨兵：开头避免空栈判断，结尾把所有柱子逼出栈
    stack = [0]
    best = 0
    for i in range(1, len(hs)):
        while hs[stack[-1]] > hs[i]:
            h = hs[stack.pop()]
            width = i - stack[-1] - 1        # 左边界 = 新栈顶，右边界 = i
            best = max(best, h * width)
        stack.append(i)
    return best


def maximal_rectangle(matrix: List[List[str]]) -> int:
    """LC 85. 每行做柱状图，套 LC 84。O(mn) / O(n)。"""
    if not matrix:
        return 0
    heights = [0] * len(matrix[0])
    best = 0
    for row in matrix:
        for j, c in enumerate(row):
            heights[j] = heights[j] + 1 if c == "1" else 0
        best = max(best, largest_rectangle_area(heights))
    return best


def max_sliding_window(nums: List[int], k: int) -> List[int]:
    """LC 239. 单调递减双端队列（存下标）：队首是窗口最大。O(n) / O(k)。"""
    dq: deque[int] = deque()
    out: List[int] = []
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] <= x:      # 比新来的小的永远不会当最大值了
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:                   # 队首滑出窗口
            dq.popleft()
        if i >= k - 1:
            out.append(nums[dq[0]])
    return out


def calculate(s: str) -> int:
    """LC 227. 基本计算器 II：栈存"带符号的项"，乘除立即与栈顶结算。O(n) / O(n)。"""
    stack: List[int] = []
    num, op = 0, "+"
    for i, ch in enumerate(s + "+"):         # 末尾补一个运算符，把最后一个数收进栈
        if ch.isdigit():
            num = num * 10 + int(ch)
        elif ch in "+-*/":
            if op == "+":
                stack.append(num)
            elif op == "-":
                stack.append(-num)
            elif op == "*":
                stack.append(stack.pop() * num)
            else:
                stack.append(int(stack.pop() / num))
            num, op = 0, ch
    return sum(stack)


def trap_stack(height: List[int]) -> int:
    """LC 42 的单调栈解：按"层"接水，与 02 篇的双指针对照。O(n) / O(n)。"""
    stack: List[int] = []
    water = 0
    for i, h in enumerate(height):
        while stack and height[stack[-1]] < h:
            bottom = height[stack.pop()]
            if not stack:
                break
            left = stack[-1]
            water += (min(height[left], h) - bottom) * (i - left - 1)
        stack.append(i)
    return water


class MinStack:
    """LC 155. 辅助栈同步存当前最小。全部 O(1)。"""

    def __init__(self) -> None:
        self.stack: List[int] = []
        self.mins: List[int] = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        self.mins.append(min(val, self.mins[-1]) if self.mins else val)

    def pop(self) -> None:
        self.stack.pop()
        self.mins.pop()

    def top(self) -> int:
        return self.stack[-1]

    def get_min(self) -> int:
        return self.mins[-1]


class Tests(unittest.TestCase):
    def test_is_valid(self):
        self.assertTrue(is_valid("()[]{}"))
        self.assertFalse(is_valid("(]"))
        self.assertFalse(is_valid("("))
        self.assertFalse(is_valid(")"))

    def test_eval_rpn(self):
        self.assertEqual(eval_rpn(["2", "1", "+", "3", "*"]), 9)
        self.assertEqual(eval_rpn(["4", "13", "5", "/", "+"]), 6)
        self.assertEqual(eval_rpn(["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]), 22)

    def test_decode_string(self):
        self.assertEqual(decode_string("3[a]2[bc]"), "aaabcbc")
        self.assertEqual(decode_string("3[a2[c]]"), "accaccacc")
        self.assertEqual(decode_string("2[abc]3[cd]ef"), "abcabccdcdcdef")

    def test_daily_temperatures(self):
        self.assertEqual(daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]), [1, 1, 4, 2, 1, 1, 0, 0])

    def test_largest_rectangle_area(self):
        self.assertEqual(largest_rectangle_area([2, 1, 5, 6, 2, 3]), 10)
        self.assertEqual(largest_rectangle_area([2, 4]), 4)
        self.assertEqual(largest_rectangle_area([1]), 1)

    def test_maximal_rectangle(self):
        m = [list("10100"), list("10111"), list("11111"), list("10010")]
        self.assertEqual(maximal_rectangle(m), 6)

    def test_max_sliding_window(self):
        self.assertEqual(max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3), [3, 3, 5, 5, 6, 7])
        self.assertEqual(max_sliding_window([1], 1), [1])

    def test_calculate(self):
        self.assertEqual(calculate("3+2*2"), 7)
        self.assertEqual(calculate(" 3/2 "), 1)
        self.assertEqual(calculate(" 3+5 / 2 "), 5)
        self.assertEqual(calculate("14-3/2"), 13)

    def test_trap_stack(self):
        self.assertEqual(trap_stack([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]), 6)

    def test_min_stack(self):
        st = MinStack()
        st.push(-2); st.push(0); st.push(-3)
        self.assertEqual(st.get_min(), -3)
        st.pop()
        self.assertEqual(st.top(), 0)
        self.assertEqual(st.get_min(), -2)


if __name__ == "__main__":
    unittest.main(verbosity=1)
