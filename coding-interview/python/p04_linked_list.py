"""面试手撕代码（04）：链表 —— 配套解法与测试。

https://arganzheng.life/coding-interview-linked-list.html
"""
from __future__ import annotations

import heapq
import unittest
from typing import List, Optional


class ListNode:
    def __init__(self, val: int = 0, next: "Optional[ListNode]" = None):
        self.val = val
        self.next = next

    def __lt__(self, other: "ListNode") -> bool:  # 让节点能直接进 heapq
        return self.val < other.val


def build(values: List[int]) -> Optional[ListNode]:
    dummy = ListNode()
    cur = dummy
    for v in values:
        cur.next = ListNode(v)
        cur = cur.next
    return dummy.next


def to_list(head: Optional[ListNode]) -> List[int]:
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out


def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """LC 206. 三指针迭代。O(n) / O(1)。"""
    prev, cur = None, head
    while cur:
        cur.next, prev, cur = prev, cur, cur.next   # 右侧先整体求值，再依次赋值
    return prev


def reverse_list_recursive(head: Optional[ListNode]) -> Optional[ListNode]:
    """LC 206 递归版：先反转后面，再把自己接到尾巴上。O(n) / O(n) 栈。"""
    if not head or not head.next:
        return head
    new_head = reverse_list_recursive(head.next)
    head.next.next = head
    head.next = None
    return new_head


def reverse_between(head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
    """LC 92. 哑节点定位 pre，然后做 right-left 次"头插"。O(n) / O(1)。"""
    dummy = ListNode(0, head)
    pre = dummy
    for _ in range(left - 1):
        pre = pre.next
    cur = pre.next
    for _ in range(right - left):
        nxt = cur.next                 # 把 nxt 摘下来，插到 pre 后面
        cur.next = nxt.next
        nxt.next = pre.next
        pre.next = nxt
    return dummy.next


def reverse_k_group(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    """LC 25. 每次先探 k 个，够则反转这一段并接回。O(n) / O(1)。"""
    dummy = ListNode(0, head)
    group_prev = dummy
    while True:
        kth = group_prev
        for _ in range(k):
            kth = kth.next
            if not kth:
                return dummy.next
        group_next = kth.next
        prev, cur = group_next, group_prev.next     # 反转段内 k 个，尾巴直接指向下一组
        while cur is not group_next:
            cur.next, prev, cur = prev, cur, cur.next
        tmp = group_prev.next                       # 原段头，反转后成为段尾
        group_prev.next = kth
        group_prev = tmp


def middle_node(head: ListNode) -> ListNode:
    """LC 876. 快慢指针：偶数长度返回第二个中点。O(n) / O(1)。"""
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
    return slow


def has_cycle(head: Optional[ListNode]) -> bool:
    """LC 141. 快慢指针相遇即有环。O(n) / O(1)。"""
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow is fast:
            return True
    return False


def detect_cycle(head: Optional[ListNode]) -> Optional[ListNode]:
    """LC 142. 相遇后一指针回头，同速再走必在入口相遇（推导见正文）。O(n) / O(1)。"""
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow is fast:
            p = head
            while p is not slow:
                p, slow = p.next, slow.next
            return p
    return None


def get_intersection_node(a: Optional[ListNode], b: Optional[ListNode]) -> Optional[ListNode]:
    """LC 160. 两指针各走 A+B 长度，第二轮对齐。O(m+n) / O(1)。"""
    p, q = a, b
    while p is not q:
        p = p.next if p else b
        q = q.next if q else a
    return p


def merge_two_lists(a: Optional[ListNode], b: Optional[ListNode]) -> Optional[ListNode]:
    """LC 21. 哑节点 + 尾指针。O(m+n) / O(1)。"""
    dummy = tail = ListNode()
    while a and b:
        if a.val <= b.val:
            tail.next, a = a, a.next
        else:
            tail.next, b = b, b.next
        tail = tail.next
    tail.next = a or b
    return dummy.next


def merge_k_lists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """LC 23. 最小堆存 (val, idx, node)：O(N log k) / O(k)。"""
    heap = [(node.val, i, node) for i, node in enumerate(lists) if node]
    heapq.heapify(heap)
    dummy = tail = ListNode()
    while heap:
        _, i, node = heapq.heappop(heap)
        tail.next = node
        tail = node
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    return dummy.next


def remove_nth_from_end(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    """LC 19. 快指针先走 n+1 步，两指针同行；哑节点处理删头。O(L) / O(1)。"""
    dummy = ListNode(0, head)
    fast = slow = dummy
    for _ in range(n + 1):
        fast = fast.next
    while fast:
        fast, slow = fast.next, slow.next
    slow.next = slow.next.next
    return dummy.next


def sort_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """LC 148. 自顶向下归并：找中点断开、递归、合并。O(n log n) / O(log n) 栈。"""
    if not head or not head.next:
        return head
    slow, fast = head, head.next          # fast 先一步：偶数长度时 slow 落在前半的末尾
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
    mid, slow.next = slow.next, None
    return merge_two_lists(sort_list(head), sort_list(mid))


def is_palindrome(head: Optional[ListNode]) -> bool:
    """LC 234. 找中点、反转后半、逐一比较。O(n) / O(1)。"""
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
    second = reverse_list(slow)
    p, q = head, second
    ok = True
    while q:
        if p.val != q.val:
            ok = False
            break
        p, q = p.next, q.next
    reverse_list(second)                  # 恢复原链表（面试加分项）
    return ok


class RandomNode:
    def __init__(self, x: int, next: "Optional[RandomNode]" = None, random: "Optional[RandomNode]" = None):
        self.val, self.next, self.random = x, next, random


def copy_random_list(head: Optional[RandomNode]) -> Optional[RandomNode]:
    """LC 138. 哈希映射 old -> new 两趟；O(1) 空间版是"交织节点"，见正文。O(n) / O(n)。"""
    if not head:
        return None
    mapping: dict[RandomNode, RandomNode] = {}
    cur = head
    while cur:
        mapping[cur] = RandomNode(cur.val)
        cur = cur.next
    cur = head
    while cur:
        mapping[cur].next = mapping.get(cur.next)
        mapping[cur].random = mapping.get(cur.random)
        cur = cur.next
    return mapping[head]


class Tests(unittest.TestCase):
    def test_reverse(self):
        self.assertEqual(to_list(reverse_list(build([1, 2, 3, 4, 5]))), [5, 4, 3, 2, 1])
        self.assertEqual(to_list(reverse_list_recursive(build([1, 2]))), [2, 1])
        self.assertIsNone(reverse_list(None))

    def test_reverse_between(self):
        self.assertEqual(to_list(reverse_between(build([1, 2, 3, 4, 5]), 2, 4)), [1, 4, 3, 2, 5])
        self.assertEqual(to_list(reverse_between(build([5]), 1, 1)), [5])

    def test_reverse_k_group(self):
        self.assertEqual(to_list(reverse_k_group(build([1, 2, 3, 4, 5]), 2)), [2, 1, 4, 3, 5])
        self.assertEqual(to_list(reverse_k_group(build([1, 2, 3, 4, 5]), 3)), [3, 2, 1, 4, 5])

    def test_middle_and_cycle(self):
        self.assertEqual(middle_node(build([1, 2, 3, 4, 5, 6])).val, 4)
        head = build([3, 2, 0, -4])
        self.assertFalse(has_cycle(head))
        head.next.next.next.next = head.next    # -4 -> 2
        self.assertTrue(has_cycle(head))
        self.assertIs(detect_cycle(head), head.next)

    def test_intersection(self):
        common = build([8, 4, 5])
        a = build([4, 1]); a.next.next = common
        b = build([5, 6, 1]); b.next.next.next = common
        self.assertIs(get_intersection_node(a, b), common)
        self.assertIsNone(get_intersection_node(build([1]), build([2])))

    def test_merge(self):
        self.assertEqual(to_list(merge_two_lists(build([1, 2, 4]), build([1, 3, 4]))), [1, 1, 2, 3, 4, 4])
        self.assertEqual(to_list(merge_k_lists([build([1, 4, 5]), build([1, 3, 4]), build([2, 6])])), [1, 1, 2, 3, 4, 4, 5, 6])
        self.assertIsNone(merge_k_lists([]))

    def test_remove_nth(self):
        self.assertEqual(to_list(remove_nth_from_end(build([1, 2, 3, 4, 5]), 2)), [1, 2, 3, 5])
        self.assertEqual(to_list(remove_nth_from_end(build([1]), 1)), [])

    def test_sort(self):
        self.assertEqual(to_list(sort_list(build([4, 2, 1, 3]))), [1, 2, 3, 4])
        self.assertEqual(to_list(sort_list(build([-1, 5, 3, 4, 0]))), [-1, 0, 3, 4, 5])

    def test_palindrome(self):
        h = build([1, 2, 2, 1])
        self.assertTrue(is_palindrome(h))
        self.assertEqual(to_list(h), [1, 2, 2, 1])
        self.assertFalse(is_palindrome(build([1, 2])))

    def test_copy_random(self):
        a, b, c = RandomNode(7), RandomNode(13), RandomNode(11)
        a.next, b.next = b, c
        b.random, c.random = a, a
        h = copy_random_list(a)
        self.assertEqual([h.val, h.next.val, h.next.next.val], [7, 13, 11])
        self.assertIsNot(h.next.random, a)
        self.assertIs(h.next.random, h)


if __name__ == "__main__":
    unittest.main(verbosity=1)
