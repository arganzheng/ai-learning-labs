import java.util.*;

/** 面试手撕代码（11）：动态规划（一）线性与二维。运行：java -ea P11DpLinearGrid */
public class P11DpLinearGrid {

    /** LC 70. */
    static int climbStairs(int n) {
        int a = 1, b = 1;
        for (int i = 1; i < n; i++) { int t = a + b; a = b; b = t; }
        return b;
    }

    /** LC 198. */
    static int rob(int[] nums) {
        int prev2 = 0, prev1 = 0;
        for (int x : nums) { int cur = Math.max(prev1, prev2 + x); prev2 = prev1; prev1 = cur; }
        return prev1;
    }

    /** LC 213. */
    static int robCircular(int[] nums) {
        if (nums.length == 1) return nums[0];
        return Math.max(rob(Arrays.copyOfRange(nums, 1, nums.length)), rob(Arrays.copyOfRange(nums, 0, nums.length - 1)));
    }

    /** LC 322. */
    static int coinChange(int[] coins, int amount) {
        int INF = amount + 1;
        int[] f = new int[amount + 1];
        Arrays.fill(f, INF);
        f[0] = 0;
        for (int a = 1; a <= amount; a++)
            for (int c : coins) if (c <= a) f[a] = Math.min(f[a], f[a - c] + 1);
        return f[amount] == INF ? -1 : f[amount];
    }

    /** LC 300. O(n log n) patience。 */
    static int lengthOfLIS(int[] nums) {
        int[] tails = new int[nums.length];
        int size = 0;
        for (int x : nums) {
            int lo = 0, hi = size;
            while (lo < hi) { int mid = (lo + hi) >>> 1; if (tails[mid] < x) lo = mid + 1; else hi = mid; }
            tails[lo] = x;
            if (lo == size) size++;
        }
        return size;
    }

    /** LC 53. Kadane。 */
    static int maxSubArray(int[] nums) {
        int best = nums[0], cur = nums[0];
        for (int i = 1; i < nums.length; i++) { cur = Math.max(nums[i], cur + nums[i]); best = Math.max(best, cur); }
        return best;
    }

    /** LC 152. */
    static int maxProduct(int[] nums) {
        int best = nums[0], curMax = nums[0], curMin = nums[0];
        for (int i = 1; i < nums.length; i++) {
            int x = nums[i];
            int mx = Math.max(x, Math.max(curMax * x, curMin * x));
            int mn = Math.min(x, Math.min(curMax * x, curMin * x));
            curMax = mx; curMin = mn;
            best = Math.max(best, curMax);
        }
        return best;
    }

    /** LC 139. */
    static boolean wordBreak(String s, List<String> wordDict) {
        Set<String> words = new HashSet<>(wordDict);
        boolean[] f = new boolean[s.length() + 1];
        f[0] = true;
        for (int i = 1; i <= s.length(); i++)
            for (int j = 0; j < i; j++)
                if (f[j] && words.contains(s.substring(j, i))) { f[i] = true; break; }
        return f[s.length()];
    }

    /** LC 62. */
    static int uniquePaths(int m, int n) {
        int[] f = new int[n];
        Arrays.fill(f, 1);
        for (int i = 1; i < m; i++) for (int j = 1; j < n; j++) f[j] += f[j - 1];
        return f[n - 1];
    }

    /** LC 64. 原地。 */
    static int minPathSum(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) {
                if (i == 0 && j == 0) continue;
                int up = i > 0 ? grid[i - 1][j] : Integer.MAX_VALUE, left = j > 0 ? grid[i][j - 1] : Integer.MAX_VALUE;
                grid[i][j] += Math.min(up, left);
            }
        return grid[m - 1][n - 1];
    }

    /** LC 221. */
    static int maximalSquare(char[][] matrix) {
        int m = matrix.length, n = matrix[0].length, best = 0;
        int[][] f = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++)
            for (int j = 1; j <= n; j++)
                if (matrix[i - 1][j - 1] == '1') {
                    f[i][j] = 1 + Math.min(f[i - 1][j - 1], Math.min(f[i - 1][j], f[i][j - 1]));
                    best = Math.max(best, f[i][j]);
                }
        return best * best;
    }

    /** LC 1143. */
    static int longestCommonSubsequence(String a, String b) {
        int m = a.length(), n = b.length();
        int[][] f = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++)
            for (int j = 1; j <= n; j++)
                f[i][j] = a.charAt(i - 1) == b.charAt(j - 1) ? f[i - 1][j - 1] + 1 : Math.max(f[i - 1][j], f[i][j - 1]);
        return f[m][n];
    }

    /** LC 72. */
    static int minDistance(String a, String b) {
        int m = a.length(), n = b.length();
        int[][] f = new int[m + 1][n + 1];
        for (int i = 0; i <= m; i++) f[i][0] = i;
        for (int j = 0; j <= n; j++) f[0][j] = j;
        for (int i = 1; i <= m; i++)
            for (int j = 1; j <= n; j++)
                f[i][j] = a.charAt(i - 1) == b.charAt(j - 1) ? f[i - 1][j - 1]
                        : 1 + Math.min(f[i - 1][j - 1], Math.min(f[i - 1][j], f[i][j - 1]));
        return f[m][n];
    }

    /** LC 91. */
    static int numDecodings(String s) {
        int prev2 = 1, prev1 = s.charAt(0) != '0' ? 1 : 0;
        for (int i = 2; i <= s.length(); i++) {
            int cur = 0;
            if (s.charAt(i - 1) != '0') cur += prev1;
            int two = Integer.parseInt(s.substring(i - 2, i));
            if (two >= 10 && two <= 26) cur += prev2;
            prev2 = prev1; prev1 = cur;
        }
        return prev1;
    }

    public static void main(String[] args) {
        assert climbStairs(5) == 8;
        assert rob(new int[]{2, 7, 9, 3, 1}) == 12 && robCircular(new int[]{1, 2, 3, 1}) == 4;
        assert coinChange(new int[]{1, 2, 5}, 11) == 3 && coinChange(new int[]{2}, 3) == -1;
        assert lengthOfLIS(new int[]{10, 9, 2, 5, 3, 7, 101, 18}) == 4 && lengthOfLIS(new int[]{7, 7, 7}) == 1;
        assert maxSubArray(new int[]{-2, 1, -3, 4, -1, 2, 1, -5, 4}) == 6;
        assert maxProduct(new int[]{-2, 3, -4}) == 24 && maxProduct(new int[]{-2, 0, -1}) == 0;
        assert wordBreak("leetcode", List.of("leet", "code")) && !wordBreak("catsandog", List.of("cats", "dog", "sand", "and", "cat"));
        assert uniquePaths(3, 7) == 28;
        assert minPathSum(new int[][]{{1, 3, 1}, {1, 5, 1}, {4, 2, 1}}) == 7;
        assert maximalSquare(new char[][]{"10100".toCharArray(), "10111".toCharArray(), "11111".toCharArray(), "10010".toCharArray()}) == 4;
        assert longestCommonSubsequence("abcde", "ace") == 3;
        assert minDistance("horse", "ros") == 3 && minDistance("intention", "execution") == 5;
        assert numDecodings("226") == 3 && numDecodings("06") == 0;
        System.out.println("P11DpLinearGrid OK");
    }
}
