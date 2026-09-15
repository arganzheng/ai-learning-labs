"""面试手撕代码（05）：二叉树 —— 配套解法与测试。

https://arganzheng.life/coding-interview-binary-tree.html
"""
from __future__ import annotations

import unittest
from collections import defaultdict, deque
from typing import List, Optional


class TreeNode:
    def __init__(self, val: int = 0, left: "Optional[TreeNode]" = None, right: "Optional[TreeNode]" = None):
        self.val, self.left, self.right = val, left, right


def build(values: List[Optional[int]]) -> Optional[TreeNode]:
    """LeetCode 风格的层序数组（None 为空）建树。"""
    if not values or values[0] is None:
        return None
    root = TreeNode(values[0])
    q = deque([root])
    i = 1
    while q and i < len(values):
        node = q.popleft()
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i]); q.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i]); q.append(node.right)
        i += 1
    return root


def inorder_iterative(root: Optional[TreeNode]) -> List[int]:
    """LC 94. 迭代中序：一路向左压栈，弹出即访问，再转右。O(n) / O(h)。"""
    out, stack, cur = [], [], root
    while cur or stack:
        while cur:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        out.append(cur.val)
        cur = cur.right
    return out


def preorder_iterative(root: Optional[TreeNode]) -> List[int]:
    """LC 144. 栈：先压右再压左。"""
    out, stack = [], [root] if root else []
    while stack:
        node = stack.pop()
        out.append(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return out


def level_order(root: Optional[TreeNode]) -> List[List[int]]:
    """LC 102. BFS 按层：每轮先记下队列长度。O(n) / O(w)。"""
    out: List[List[int]] = []
    q = deque([root] if root else [])
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        out.append(level)
    return out


def right_side_view(root: Optional[TreeNode]) -> List[int]:
    """LC 199. 每层最后一个。"""
    return [level[-1] for level in level_order(root)]


def max_depth(root: Optional[TreeNode]) -> int:
    """LC 104. 后序：1 + max(左, 右)。"""
    return 0 if not root else 1 + max(max_depth(root.left), max_depth(root.right))


def invert_tree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """LC 226."""
    if root:
        root.left, root.right = invert_tree(root.right), invert_tree(root.left)
    return root


def is_symmetric(root: Optional[TreeNode]) -> bool:
    """LC 101. 两棵子树镜像比较。"""
    def mirror(a, b):
        if not a and not b:
            return True
        if not a or not b or a.val != b.val:
            return False
        return mirror(a.left, b.right) and mirror(a.right, b.left)
    return mirror(root.left, root.right) if root else True


def diameter_of_binary_tree(root: Optional[TreeNode]) -> int:
    """LC 543. 后序返回"向下最长链"，路径长度在合并处更新。O(n) / O(h)。"""
    best = 0

    def depth(node):
        nonlocal best
        if not node:
            return 0
        l, r = depth(node.left), depth(node.right)
        best = max(best, l + r)
        return 1 + max(l, r)

    depth(root)
    return best


def max_path_sum(root: Optional[TreeNode]) -> int:
    """LC 124. 与 543 同一模板：负贡献截断为 0。O(n) / O(h)。"""
    best = float("-inf")

    def gain(node):
        nonlocal best
        if not node:
            return 0
        l = max(gain(node.left), 0)
        r = max(gain(node.right), 0)
        best = max(best, node.val + l + r)
        return node.val + max(l, r)

    gain(root)
    return best


def lowest_common_ancestor(root: TreeNode, p: TreeNode, q: TreeNode) -> Optional[TreeNode]:
    """LC 236. 后序：左右各返回找到的节点；两边都有则当前是 LCA。O(n) / O(h)。"""
    if not root or root is p or root is q:
        return root
    l = lowest_common_ancestor(root.left, p, q)
    r = lowest_common_ancestor(root.right, p, q)
    if l and r:
        return root
    return l or r


def build_tree(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    """LC 105. 前序第一个是根，在中序里定位切分左右；哈希 O(1) 查位置。O(n) / O(n)。"""
    pos = {v: i for i, v in enumerate(inorder)}
    pre_idx = [0]

    def rec(lo: int, hi: int) -> Optional[TreeNode]:   # 中序区间 [lo, hi)
        if lo >= hi:
            return None
        val = preorder[pre_idx[0]]
        pre_idx[0] += 1
        node = TreeNode(val)
        mid = pos[val]
        node.left = rec(lo, mid)                        # 先建左，前序指针自然先消耗左子树
        node.right = rec(mid + 1, hi)
        return node

    return rec(0, len(inorder))


def is_valid_bst(root: Optional[TreeNode]) -> bool:
    """LC 98. 带上下界递归（不能只比较父子）。O(n) / O(h)。"""
    def check(node, lo, hi):
        if not node:
            return True
        if not (lo < node.val < hi):
            return False
        return check(node.left, lo, node.val) and check(node.right, node.val, hi)
    return check(root, float("-inf"), float("inf"))


def kth_smallest(root: Optional[TreeNode], k: int) -> int:
    """LC 230. 迭代中序数到第 k 个即停。O(h + k)。"""
    stack, cur = [], root
    while True:
        while cur:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        k -= 1
        if k == 0:
            return cur.val
        cur = cur.right


def serialize(root: Optional[TreeNode]) -> str:
    """LC 297. 前序 + '#' 占空位。"""
    out: List[str] = []

    def rec(node):
        if not node:
            out.append("#")
            return
        out.append(str(node.val))
        rec(node.left)
        rec(node.right)

    rec(root)
    return ",".join(out)


def deserialize(data: str) -> Optional[TreeNode]:
    """LC 297. 用迭代器按前序还原。"""
    it = iter(data.split(","))

    def rec():
        tok = next(it)
        if tok == "#":
            return None
        node = TreeNode(int(tok))
        node.left = rec()
        node.right = rec()
        return node

    return rec()


def path_sum_iii(root: Optional[TreeNode], target: int) -> int:
    """LC 437. 树上前缀和 + 哈希（LC 560 的树版），回溯时撤销计数。O(n) / O(h)。"""
    count = defaultdict(int)
    count[0] = 1
    ans = 0

    def dfs(node, pre):
        nonlocal ans
        if not node:
            return
        pre += node.val
        ans += count[pre - target]
        count[pre] += 1
        dfs(node.left, pre)
        dfs(node.right, pre)
        count[pre] -= 1                  # 离开这条路径时撤销

    dfs(root, 0)
    return ans


def flatten(root: Optional[TreeNode]) -> None:
    """LC 114. 原地展开为前序链表：把左子树接到右边、原右子树接到左子树最右。O(n) / O(1)。"""
    cur = root
    while cur:
        if cur.left:
            pre = cur.left
            while pre.right:
                pre = pre.right
            pre.right = cur.right
            cur.right, cur.left = cur.left, None
        cur = cur.right


class Tests(unittest.TestCase):
    def setUp(self):
        self.t = build([3, 9, 20, None, None, 15, 7])

    def test_traversals(self):
        self.assertEqual(inorder_iterative(self.t), [9, 3, 15, 20, 7])
        self.assertEqual(preorder_iterative(self.t), [3, 9, 20, 15, 7])
        self.assertEqual(level_order(self.t), [[3], [9, 20], [15, 7]])
        self.assertEqual(right_side_view(build([1, 2, 3, None, 5, None, 4])), [1, 3, 4])

    def test_basic(self):
        self.assertEqual(max_depth(self.t), 3)
        self.assertEqual(level_order(invert_tree(build([4, 2, 7, 1, 3, 6, 9]))), [[4], [7, 2], [9, 6, 3, 1]])
        self.assertTrue(is_symmetric(build([1, 2, 2, 3, 4, 4, 3])))
        self.assertFalse(is_symmetric(build([1, 2, 2, None, 3, None, 3])))

    def test_paths(self):
        self.assertEqual(diameter_of_binary_tree(build([1, 2, 3, 4, 5])), 3)
        self.assertEqual(max_path_sum(build([1, 2, 3])), 6)
        self.assertEqual(max_path_sum(build([-10, 9, 20, None, None, 15, 7])), 42)
        self.assertEqual(max_path_sum(build([-3])), -3)

    def test_lca(self):
        t = build([3, 5, 1, 6, 2, 0, 8, None, None, 7, 4])
        p, q = t.left, t.right                    # 5, 1
        self.assertIs(lowest_common_ancestor(t, p, q), t)
        self.assertIs(lowest_common_ancestor(t, p, t.left.right.right), p)   # 5 与 4 -> 5

    def test_build(self):
        t = build_tree([3, 9, 20, 15, 7], [9, 3, 15, 20, 7])
        self.assertEqual(level_order(t), [[3], [9, 20], [15, 7]])

    def test_bst(self):
        self.assertTrue(is_valid_bst(build([2, 1, 3])))
        self.assertFalse(is_valid_bst(build([5, 1, 4, None, None, 3, 6])))
        self.assertFalse(is_valid_bst(build([5, 4, 6, None, None, 3, 7])))    # 3 在右子树里但小于 5
        self.assertEqual(kth_smallest(build([5, 3, 6, 2, 4, None, None, 1]), 3), 3)

    def test_serialize(self):
        s = serialize(self.t)
        self.assertEqual(s, "3,9,#,#,20,15,#,#,7,#,#")
        self.assertEqual(level_order(deserialize(s)), [[3], [9, 20], [15, 7]])
        self.assertIsNone(deserialize(serialize(None)))

    def test_path_sum_iii(self):
        self.assertEqual(path_sum_iii(build([10, 5, -3, 3, 2, None, 11, 3, -2, None, 1]), 8), 3)

    def test_flatten(self):
        t = build([1, 2, 5, 3, 4, None, 6])
        flatten(t)
        vals, cur = [], t
        while cur:
            self.assertIsNone(cur.left)
            vals.append(cur.val)
            cur = cur.right
        self.assertEqual(vals, [1, 2, 3, 4, 5, 6])


if __name__ == "__main__":
    unittest.main(verbosity=1)
