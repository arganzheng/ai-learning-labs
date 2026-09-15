import java.util.*;

/** 面试手撕代码（05）：二叉树。运行：java -ea P05BinaryTree */
public class P05BinaryTree {

    static class TreeNode {
        int val; TreeNode left, right;
        TreeNode(int v) { val = v; }
    }

    /** LeetCode 风格层序数组建树，null 为空。 */
    static TreeNode build(Integer... vals) {
        if (vals.length == 0 || vals[0] == null) return null;
        TreeNode root = new TreeNode(vals[0]);
        Deque<TreeNode> q = new ArrayDeque<>(List.of(root));
        int i = 1;
        while (!q.isEmpty() && i < vals.length) {
            TreeNode n = q.poll();
            if (i < vals.length && vals[i] != null) { n.left = new TreeNode(vals[i]); q.add(n.left); }
            i++;
            if (i < vals.length && vals[i] != null) { n.right = new TreeNode(vals[i]); q.add(n.right); }
            i++;
        }
        return root;
    }

    /** LC 94. 迭代中序。 */
    static List<Integer> inorderIterative(TreeNode root) {
        List<Integer> out = new ArrayList<>();
        Deque<TreeNode> stack = new ArrayDeque<>();
        TreeNode cur = root;
        while (cur != null || !stack.isEmpty()) {
            while (cur != null) { stack.push(cur); cur = cur.left; }
            cur = stack.pop();
            out.add(cur.val);
            cur = cur.right;
        }
        return out;
    }

    /** LC 102. 按层 BFS。 */
    static List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> out = new ArrayList<>();
        if (root == null) return out;
        Deque<TreeNode> q = new ArrayDeque<>(List.of(root));
        while (!q.isEmpty()) {
            List<Integer> level = new ArrayList<>();
            for (int n = q.size(); n > 0; n--) {
                TreeNode node = q.poll();
                level.add(node.val);
                if (node.left != null) q.add(node.left);
                if (node.right != null) q.add(node.right);
            }
            out.add(level);
        }
        return out;
    }

    /** LC 104. */
    static int maxDepth(TreeNode r) { return r == null ? 0 : 1 + Math.max(maxDepth(r.left), maxDepth(r.right)); }

    /** LC 226. */
    static TreeNode invertTree(TreeNode r) {
        if (r == null) return null;
        TreeNode l = invertTree(r.left);
        r.left = invertTree(r.right);
        r.right = l;
        return r;
    }

    /** LC 101. */
    static boolean isSymmetric(TreeNode r) { return r == null || mirror(r.left, r.right); }
    private static boolean mirror(TreeNode a, TreeNode b) {
        if (a == null || b == null) return a == b;
        return a.val == b.val && mirror(a.left, b.right) && mirror(a.right, b.left);
    }

    /** LC 543. 后序返回向下最长链，路径在合并处更新。 */
    static int diameter;
    static int diameterOfBinaryTree(TreeNode root) { diameter = 0; depth(root); return diameter; }
    private static int depth(TreeNode n) {
        if (n == null) return 0;
        int l = depth(n.left), r = depth(n.right);
        diameter = Math.max(diameter, l + r);
        return 1 + Math.max(l, r);
    }

    /** LC 124. 同一模板，负贡献截断为 0。 */
    static int best;
    static int maxPathSum(TreeNode root) { best = Integer.MIN_VALUE; gain(root); return best; }
    private static int gain(TreeNode n) {
        if (n == null) return 0;
        int l = Math.max(gain(n.left), 0), r = Math.max(gain(n.right), 0);
        best = Math.max(best, n.val + l + r);
        return n.val + Math.max(l, r);
    }

    /** LC 236. */
    static TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) return root;
        TreeNode l = lowestCommonAncestor(root.left, p, q), r = lowestCommonAncestor(root.right, p, q);
        if (l != null && r != null) return root;
        return l != null ? l : r;
    }

    /** LC 105. 前序 + 中序建树。 */
    static int preIdx;
    static TreeNode buildTree(int[] preorder, int[] inorder) {
        Map<Integer, Integer> pos = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) pos.put(inorder[i], i);
        preIdx = 0;
        return rec(preorder, pos, 0, inorder.length);
    }
    private static TreeNode rec(int[] pre, Map<Integer, Integer> pos, int lo, int hi) {
        if (lo >= hi) return null;
        TreeNode n = new TreeNode(pre[preIdx++]);
        int mid = pos.get(n.val);
        n.left = rec(pre, pos, lo, mid);
        n.right = rec(pre, pos, mid + 1, hi);
        return n;
    }

    /** LC 98. 上下界用 Long 避开 Integer 极值。 */
    static boolean isValidBST(TreeNode r) { return check(r, Long.MIN_VALUE, Long.MAX_VALUE); }
    private static boolean check(TreeNode n, long lo, long hi) {
        if (n == null) return true;
        if (n.val <= lo || n.val >= hi) return false;
        return check(n.left, lo, n.val) && check(n.right, n.val, hi);
    }

    /** LC 230. */
    static int kthSmallest(TreeNode root, int k) {
        Deque<TreeNode> stack = new ArrayDeque<>();
        TreeNode cur = root;
        while (true) {
            while (cur != null) { stack.push(cur); cur = cur.left; }
            cur = stack.pop();
            if (--k == 0) return cur.val;
            cur = cur.right;
        }
    }

    /** LC 297. 前序 + '#'。 */
    static String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        ser(root, sb);
        return sb.substring(0, sb.length() - 1);
    }
    private static void ser(TreeNode n, StringBuilder sb) {
        if (n == null) { sb.append("#,"); return; }
        sb.append(n.val).append(',');
        ser(n.left, sb); ser(n.right, sb);
    }
    static TreeNode deserialize(String data) {
        return des(new ArrayDeque<>(Arrays.asList(data.split(","))));
    }
    private static TreeNode des(Deque<String> toks) {
        String t = toks.poll();
        if (t.equals("#")) return null;
        TreeNode n = new TreeNode(Integer.parseInt(t));
        n.left = des(toks); n.right = des(toks);
        return n;
    }

    /** LC 437. 树上前缀和 + 哈希，回溯撤销。 */
    static int pathSumIII(TreeNode root, int target) {
        Map<Long, Integer> count = new HashMap<>();
        count.put(0L, 1);
        return dfs(root, 0L, target, count);
    }
    private static int dfs(TreeNode n, long pre, int target, Map<Long, Integer> count) {
        if (n == null) return 0;
        pre += n.val;
        int ans = count.getOrDefault(pre - target, 0);
        count.merge(pre, 1, Integer::sum);
        ans += dfs(n.left, pre, target, count) + dfs(n.right, pre, target, count);
        count.merge(pre, -1, Integer::sum);
        return ans;
    }

    /** LC 114. 原地展开。 */
    static void flatten(TreeNode root) {
        for (TreeNode cur = root; cur != null; cur = cur.right) {
            if (cur.left != null) {
                TreeNode pre = cur.left;
                while (pre.right != null) pre = pre.right;
                pre.right = cur.right;
                cur.right = cur.left;
                cur.left = null;
            }
        }
    }

    public static void main(String[] args) {
        TreeNode t = build(3, 9, 20, null, null, 15, 7);
        assert inorderIterative(t).equals(List.of(9, 3, 15, 20, 7));
        assert levelOrder(t).equals(List.of(List.of(3), List.of(9, 20), List.of(15, 7)));
        assert maxDepth(t) == 3;
        assert isSymmetric(build(1, 2, 2, 3, 4, 4, 3));
        assert levelOrder(invertTree(build(4, 2, 7, 1, 3, 6, 9))).get(1).equals(List.of(7, 2));
        assert diameterOfBinaryTree(build(1, 2, 3, 4, 5)) == 3;
        assert maxPathSum(build(-10, 9, 20, null, null, 15, 7)) == 42;
        TreeNode lca = build(3, 5, 1, 6, 2, 0, 8, null, null, 7, 4);
        assert lowestCommonAncestor(lca, lca.left, lca.right) == lca;
        assert lowestCommonAncestor(lca, lca.left, lca.left.right.right) == lca.left;
        assert levelOrder(buildTree(new int[]{3, 9, 20, 15, 7}, new int[]{9, 3, 15, 20, 7})).equals(levelOrder(t));
        assert isValidBST(build(2, 1, 3)) && !isValidBST(build(5, 4, 6, null, null, 3, 7));
        assert isValidBST(build(Integer.MAX_VALUE));
        assert kthSmallest(build(5, 3, 6, 2, 4, null, null, 1), 3) == 3;
        assert serialize(t).equals("3,9,#,#,20,15,#,#,7,#,#");
        assert levelOrder(deserialize(serialize(t))).equals(levelOrder(t));
        assert pathSumIII(build(10, 5, -3, 3, 2, null, 11, 3, -2, null, 1), 8) == 3;
        TreeNode f = build(1, 2, 5, 3, 4, null, 6);
        flatten(f);
        List<Integer> vals = new ArrayList<>();
        for (TreeNode c = f; c != null; c = c.right) { assert c.left == null; vals.add(c.val); }
        assert vals.equals(List.of(1, 2, 3, 4, 5, 6));
        System.out.println("P05BinaryTree OK");
    }
}
