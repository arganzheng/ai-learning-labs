import java.util.*;

/** 面试手撕代码（09）：回溯。运行：java -ea P09Backtracking */
public class P09Backtracking {

    /** LC 46. */
    static List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> out = new ArrayList<>();
        bt46(nums, new boolean[nums.length], new ArrayDeque<>(), out);
        return out;
    }

    private static void bt46(
            int[] nums, boolean[] used, Deque<Integer> path, List<List<Integer>> out) {
        if (path.size() == nums.length) {
            out.add(new ArrayList<>(path));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) continue;
            used[i] = true;
            path.addLast(nums[i]);
            bt46(nums, used, path, out);
            path.removeLast();
            used[i] = false;
        }
    }

    /** LC 47. 排序 + 同层去重。 */
    static List<List<Integer>> permuteUnique(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> out = new ArrayList<>();
        bt47(nums, new boolean[nums.length], new ArrayDeque<>(), out);
        return out;
    }

    private static void bt47(
            int[] nums, boolean[] used, Deque<Integer> path, List<List<Integer>> out) {
        if (path.size() == nums.length) {
            out.add(new ArrayList<>(path));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i] || (i > 0 && nums[i] == nums[i - 1] && !used[i - 1])) continue;
            used[i] = true;
            path.addLast(nums[i]);
            bt47(nums, used, path, out);
            path.removeLast();
            used[i] = false;
        }
    }

    /** LC 78. */
    static List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> out = new ArrayList<>();
        bt78(nums, 0, new ArrayDeque<>(), out);
        return out;
    }

    private static void bt78(int[] nums, int start, Deque<Integer> path, List<List<Integer>> out) {
        out.add(new ArrayList<>(path));
        for (int i = start; i < nums.length; i++) {
            path.addLast(nums[i]);
            bt78(nums, i + 1, path, out);
            path.removeLast();
        }
    }

    /** LC 90. */
    static List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> out = new ArrayList<>();
        bt90(nums, 0, new ArrayDeque<>(), out);
        return out;
    }

    private static void bt90(int[] nums, int start, Deque<Integer> path, List<List<Integer>> out) {
        out.add(new ArrayList<>(path));
        for (int i = start; i < nums.length; i++) {
            if (i > start && nums[i] == nums[i - 1]) continue;
            path.addLast(nums[i]);
            bt90(nums, i + 1, path, out);
            path.removeLast();
        }
    }

    /** LC 39. 可重复选：递归传 i。 */
    static List<List<Integer>> combinationSum(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> out = new ArrayList<>();
        bt39(candidates, 0, target, new ArrayDeque<>(), out);
        return out;
    }

    private static void bt39(
            int[] c, int start, int remain, Deque<Integer> path, List<List<Integer>> out) {
        if (remain == 0) {
            out.add(new ArrayList<>(path));
            return;
        }
        for (int i = start; i < c.length && c[i] <= remain; i++) {
            path.addLast(c[i]);
            bt39(c, i, remain - c[i], path, out);
            path.removeLast();
        }
    }

    /** LC 40. */
    static List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> out = new ArrayList<>();
        bt40(candidates, 0, target, new ArrayDeque<>(), out);
        return out;
    }

    private static void bt40(
            int[] c, int start, int remain, Deque<Integer> path, List<List<Integer>> out) {
        if (remain == 0) {
            out.add(new ArrayList<>(path));
            return;
        }
        for (int i = start; i < c.length && c[i] <= remain; i++) {
            if (i > start && c[i] == c[i - 1]) continue;
            path.addLast(c[i]);
            bt40(c, i + 1, remain - c[i], path, out);
            path.removeLast();
        }
    }

    /** LC 22. */
    static List<String> generateParenthesis(int n) {
        List<String> out = new ArrayList<>();
        bt22(n, new StringBuilder(), 0, 0, out);
        return out;
    }

    private static void bt22(int n, StringBuilder sb, int open, int close, List<String> out) {
        if (sb.length() == 2 * n) {
            out.add(sb.toString());
            return;
        }
        if (open < n) {
            sb.append('(');
            bt22(n, sb, open + 1, close, out);
            sb.deleteCharAt(sb.length() - 1);
        }
        if (close < open) {
            sb.append(')');
            bt22(n, sb, open, close + 1, out);
            sb.deleteCharAt(sb.length() - 1);
        }
    }

    /** LC 131. 切分 + 回文表。 */
    static List<List<String>> partition(String s) {
        int n = s.length();
        boolean[][] isPal = new boolean[n][n];
        for (int i = n - 1; i >= 0; i--)
            for (int j = i; j < n; j++)
                isPal[i][j] = s.charAt(i) == s.charAt(j) && (j - i < 2 || isPal[i + 1][j - 1]);
        List<List<String>> out = new ArrayList<>();
        bt131(s, 0, isPal, new ArrayDeque<>(), out);
        return out;
    }

    private static void bt131(
            String s, int start, boolean[][] isPal, Deque<String> path, List<List<String>> out) {
        if (start == s.length()) {
            out.add(new ArrayList<>(path));
            return;
        }
        for (int end = start; end < s.length(); end++) {
            if (!isPal[start][end]) continue;
            path.addLast(s.substring(start, end + 1));
            bt131(s, end + 1, isPal, path, out);
            path.removeLast();
        }
    }

    /** LC 79. */
    static boolean exist(char[][] board, String word) {
        for (int i = 0; i < board.length; i++)
            for (int j = 0; j < board[0].length; j++) if (dfs(board, word, i, j, 0)) return true;
        return false;
    }

    private static boolean dfs(char[][] b, String w, int i, int j, int k) {
        if (k == w.length()) return true;
        if (i < 0 || j < 0 || i >= b.length || j >= b[0].length || b[i][j] != w.charAt(k))
            return false;
        char tmp = b[i][j];
        b[i][j] = '#';
        boolean found =
                dfs(b, w, i + 1, j, k + 1)
                        || dfs(b, w, i - 1, j, k + 1)
                        || dfs(b, w, i, j + 1, k + 1)
                        || dfs(b, w, i, j - 1, k + 1);
        b[i][j] = tmp;
        return found;
    }

    /** LC 51. 三个 boolean 数组代替集合。 */
    static List<List<String>> solveNQueens(int n) {
        List<List<String>> out = new ArrayList<>();
        bt51(n, 0, new int[n], new boolean[n], new boolean[2 * n], new boolean[2 * n], out);
        return out;
    }

    private static void bt51(
            int n,
            int r,
            int[] queens,
            boolean[] cols,
            boolean[] d1,
            boolean[] d2,
            List<List<String>> out) {
        if (r == n) {
            List<String> board = new ArrayList<>();
            for (int c : queens) {
                char[] row = new char[n];
                Arrays.fill(row, '.');
                row[c] = 'Q';
                board.add(new String(row));
            }
            out.add(board);
            return;
        }
        for (int c = 0; c < n; c++) {
            if (cols[c] || d1[r - c + n] || d2[r + c]) continue;
            cols[c] = d1[r - c + n] = d2[r + c] = true;
            queens[r] = c;
            bt51(n, r + 1, queens, cols, d1, d2, out);
            cols[c] = d1[r - c + n] = d2[r + c] = false;
        }
    }

    public static void main(String[] args) {
        assert permute(new int[] {1, 2, 3}).size() == 6;
        assert permuteUnique(new int[] {1, 1, 2})
                .equals(List.of(List.of(1, 1, 2), List.of(1, 2, 1), List.of(2, 1, 1)));
        assert subsets(new int[] {1, 2, 3}).size() == 8;
        assert subsetsWithDup(new int[] {1, 2, 2}).size() == 6;
        assert combinationSum(new int[] {2, 3, 6, 7}, 7)
                .equals(List.of(List.of(2, 2, 3), List.of(7)));
        assert combinationSum2(new int[] {10, 1, 2, 7, 6, 1, 5}, 8).size() == 4;
        assert generateParenthesis(3).size() == 5;
        assert partition("aab").equals(List.of(List.of("a", "a", "b"), List.of("aa", "b")));
        char[][] board = {"ABCE".toCharArray(), "SFCS".toCharArray(), "ADEE".toCharArray()};
        assert exist(board, "ABCCED") && !exist(board, "ABCB");
        assert solveNQueens(4).size() == 2 && solveNQueens(8).size() == 92;
        System.out.println("P09Backtracking OK");
    }
}
