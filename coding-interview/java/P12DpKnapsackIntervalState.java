import java.util.*;

/** 面试手撕代码（12）：动态规划（二）背包、区间、状态机、树形。运行：java -ea P12DpKnapsackIntervalState */
public class P12DpKnapsackIntervalState {

    /** LC 416. 0/1 背包，容量倒序。 */
    static boolean canPartition(int[] nums) {
        int total = 0;
        for (int x : nums) total += x;
        if (total % 2 != 0) return false;
        int target = total / 2;
        boolean[] f = new boolean[target + 1];
        f[0] = true;
        for (int x : nums) for (int c = target; c >= x; c--) f[c] |= f[c - x];
        return f[target];
    }

    /** LC 494. */
    static int findTargetSumWays(int[] nums, int target) {
        int s = 0;
        for (int x : nums) s += x;
        if ((s + target) % 2 != 0 || Math.abs(target) > s) return 0;
        int cap = (s + target) / 2;
        int[] f = new int[cap + 1];
        f[0] = 1;
        for (int x : nums) for (int c = cap; c >= x; c--) f[c] += f[c - x];
        return f[cap];
    }

    /** LC 518. 完全背包，容量正序。 */
    static int change(int amount, int[] coins) {
        int[] f = new int[amount + 1];
        f[0] = 1;
        for (int c : coins) for (int a = c; a <= amount; a++) f[a] += f[a - c];
        return f[amount];
    }

    /** LC 312. 区间 DP，k 是最后戳的。 */
    static int maxCoins(int[] nums) {
        int n = nums.length + 2;
        int[] a = new int[n];
        a[0] = a[n - 1] = 1;
        System.arraycopy(nums, 0, a, 1, nums.length);
        int[][] f = new int[n][n];
        for (int len = 2; len < n; len++)
            for (int i = 0; i + len < n; i++) {
                int j = i + len;
                for (int k = i + 1; k < j; k++)
                    f[i][j] = Math.max(f[i][j], f[i][k] + f[k][j] + a[i] * a[k] * a[j]);
            }
        return f[0][n - 1];
    }

    /** LC 516. */
    static int longestPalindromeSubseq(String s) {
        int n = s.length();
        int[][] f = new int[n][n];
        for (int i = n - 1; i >= 0; i--) {
            f[i][i] = 1;
            for (int j = i + 1; j < n; j++)
                f[i][j] = s.charAt(i) == s.charAt(j) ? f[i + 1][j - 1] + 2 : Math.max(f[i + 1][j], f[i][j - 1]);
        }
        return f[0][n - 1];
    }

    /** LC 121. */
    static int maxProfitOne(int[] prices) {
        int lo = Integer.MAX_VALUE, best = 0;
        for (int p : prices) { lo = Math.min(lo, p); best = Math.max(best, p - lo); }
        return best;
    }

    /** LC 188（含 123）. hold / free 状态机。 */
    static int maxProfitK(int k, int[] prices) {
        if (prices.length == 0) return 0;
        if (k >= prices.length / 2) {
            int s = 0;
            for (int i = 1; i < prices.length; i++) s += Math.max(0, prices[i] - prices[i - 1]);
            return s;
        }
        int[] hold = new int[k + 1], free = new int[k + 1];
        Arrays.fill(hold, Integer.MIN_VALUE / 2);
        for (int p : prices)
            for (int j = k; j >= 1; j--) {
                free[j] = Math.max(free[j], hold[j] + p);
                hold[j] = Math.max(hold[j], free[j - 1] - p);
            }
        return free[k];
    }

    /** LC 309. */
    static int maxProfitCooldown(int[] prices) {
        int hold = Integer.MIN_VALUE / 2, sold = 0, rest = 0;
        for (int p : prices) {
            int nh = Math.max(hold, rest - p), ns = hold + p, nr = Math.max(rest, sold);
            hold = nh; sold = ns; rest = nr;
        }
        return Math.max(sold, rest);
    }

    /** LC 714. */
    static int maxProfitFee(int[] prices, int fee) {
        int hold = Integer.MIN_VALUE / 2, free = 0;
        for (int p : prices) {
            int nh = Math.max(hold, free - p), nf = Math.max(free, hold + p - fee);
            hold = nh; free = nf;
        }
        return free;
    }

    static class TreeNode {
        int val; TreeNode left, right;
        TreeNode(int v, TreeNode l, TreeNode r) { val = v; left = l; right = r; }
    }

    /** LC 337. 返回 {偷, 不偷}。 */
    static int robTree(TreeNode root) { int[] r = dfs(root); return Math.max(r[0], r[1]); }
    private static int[] dfs(TreeNode n) {
        if (n == null) return new int[]{0, 0};
        int[] l = dfs(n.left), r = dfs(n.right);
        return new int[]{n.val + l[1] + r[1], Math.max(l[0], l[1]) + Math.max(r[0], r[1])};
    }

    /** LC 10. 自底向上：f[i][j] = s[i:] 与 p[j:] 匹配。 */
    static boolean isMatchRegex(String s, String p) {
        int m = s.length(), n = p.length();
        boolean[][] f = new boolean[m + 1][n + 1];
        f[m][n] = true;
        for (int i = m; i >= 0; i--)
            for (int j = n - 1; j >= 0; j--) {
                boolean first = i < m && (p.charAt(j) == s.charAt(i) || p.charAt(j) == '.');
                if (j + 1 < n && p.charAt(j + 1) == '*') f[i][j] = f[i][j + 2] || (first && f[i + 1][j]);
                else f[i][j] = first && f[i + 1][j + 1];
            }
        return f[0][0];
    }

    /** LC 44. */
    static boolean isMatchWildcard(String s, String p) {
        int m = s.length(), n = p.length();
        boolean[][] f = new boolean[m + 1][n + 1];
        f[0][0] = true;
        for (int j = 1; j <= n; j++) f[0][j] = f[0][j - 1] && p.charAt(j - 1) == '*';
        for (int i = 1; i <= m; i++)
            for (int j = 1; j <= n; j++) {
                char c = p.charAt(j - 1);
                if (c == '*') f[i][j] = f[i][j - 1] || f[i - 1][j];
                else f[i][j] = f[i - 1][j - 1] && (c == '?' || c == s.charAt(i - 1));
            }
        return f[m][n];
    }

    public static void main(String[] args) {
        assert canPartition(new int[]{1, 5, 11, 5}) && !canPartition(new int[]{1, 2, 3, 5});
        assert findTargetSumWays(new int[]{1, 1, 1, 1, 1}, 3) == 5;
        assert change(5, new int[]{1, 2, 5}) == 4;
        assert maxCoins(new int[]{3, 1, 5, 8}) == 167;
        assert longestPalindromeSubseq("bbbab") == 4;
        assert maxProfitOne(new int[]{7, 1, 5, 3, 6, 4}) == 5;
        assert maxProfitK(2, new int[]{3, 3, 5, 0, 0, 3, 1, 4}) == 6 && maxProfitK(2, new int[]{3, 2, 6, 5, 0, 3}) == 7;
        assert maxProfitCooldown(new int[]{1, 2, 3, 0, 2}) == 3;
        assert maxProfitFee(new int[]{1, 3, 2, 8, 4, 9}, 2) == 8;
        TreeNode t = new TreeNode(3, new TreeNode(2, null, new TreeNode(3, null, null)), new TreeNode(3, null, new TreeNode(1, null, null)));
        assert robTree(t) == 7;
        assert isMatchRegex("aab", "c*a*b") && !isMatchRegex("mississippi", "mis*is*p*.");
        assert isMatchWildcard("adceb", "*a*b") && !isMatchWildcard("acdcb", "a*c?b");
        System.out.println("P12DpKnapsackIntervalState OK");
    }
}
