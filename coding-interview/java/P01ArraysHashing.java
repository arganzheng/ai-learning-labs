import java.util.*;

/** 面试手撕代码（01）：数组、哈希与前缀和。运行：java -ea P01ArraysHashing */
public class P01ArraysHashing {

    /** LC 1. 一遍哈希。O(n) / O(n)。 */
    static int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> seen = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            Integer j = seen.get(target - nums[i]);
            if (j != null) return new int[]{j, i};
            seen.put(nums[i], i);
        }
        return new int[0];
    }

    /** LC 560. 前缀和 + 哈希计数。O(n) / O(n)。 */
    static int subarraySum(int[] nums, int k) {
        Map<Integer, Integer> count = new HashMap<>();
        count.put(0, 1);
        int pre = 0, ans = 0;
        for (int x : nums) {
            pre += x;
            ans += count.getOrDefault(pre - k, 0);
            count.merge(pre, 1, Integer::sum);
        }
        return ans;
    }

    /** LC 128. 只从序列起点向右数。O(n) / O(n)。 */
    static int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int x : nums) set.add(x);
        int best = 0;
        for (int x : set) {
            if (set.contains(x - 1)) continue;
            int y = x;
            while (set.contains(y + 1)) y++;
            best = Math.max(best, y - x + 1);
        }
        return best;
    }

    /** LC 41. 原地哈希：值 v 放到下标 v-1。O(n) / O(1)。 */
    static int firstMissingPositive(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            while (nums[i] >= 1 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) {
                int j = nums[i] - 1, t = nums[i];
                nums[i] = nums[j];
                nums[j] = t;
            }
        }
        for (int i = 0; i < n; i++) if (nums[i] != i + 1) return i + 1;
        return n + 1;
    }

    /** LC 238. 前缀积 + 后缀积。O(n) / O(1) 额外。 */
    static int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] out = new int[n];
        out[0] = 1;
        for (int i = 1; i < n; i++) out[i] = out[i - 1] * nums[i - 1];
        int suffix = 1;
        for (int i = n - 1; i >= 0; i--) {
            out[i] *= suffix;
            suffix *= nums[i];
        }
        return out;
    }

    /** LC 1109. 差分数组。O(n + m)。 */
    static int[] corpFlightBookings(int[][] bookings, int n) {
        int[] diff = new int[n + 1];
        for (int[] b : bookings) {
            diff[b[0] - 1] += b[2];
            diff[b[1]] -= b[2];
        }
        int[] out = new int[n];
        int run = 0;
        for (int i = 0; i < n; i++) {
            run += diff[i];
            out[i] = run;
        }
        return out;
    }

    /** LC 136. 异或。 */
    static int singleNumber(int[] nums) {
        int x = 0;
        for (int v : nums) x ^= v;
        return x;
    }

    /** LC 260. 按最低位 1 分组异或。 */
    static int[] singleNumberIII(int[] nums) {
        int xor = 0;
        for (int v : nums) xor ^= v;
        int low = xor & -xor;                 // 注意：xor 为 Integer.MIN_VALUE 时 -xor 溢出，但 & 的结果仍正确
        int a = 0;
        for (int v : nums) if ((v & low) != 0) a ^= v;
        int b = xor ^ a;
        return new int[]{Math.min(a, b), Math.max(a, b)};
    }

    /** LC 287. Floyd 判环。O(n) / O(1)。 */
    static int findDuplicate(int[] nums) {
        int slow = nums[0], fast = nums[0];
        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);
        slow = nums[0];
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }

    public static void main(String[] args) {
        assert Arrays.equals(twoSum(new int[]{2, 7, 11, 15}, 9), new int[]{0, 1});
        assert subarraySum(new int[]{1, 1, 1}, 2) == 2;
        assert subarraySum(new int[]{1, -1, 0}, 0) == 3;
        assert longestConsecutive(new int[]{100, 4, 200, 1, 3, 2}) == 4;
        assert firstMissingPositive(new int[]{3, 4, -1, 1}) == 2;
        assert firstMissingPositive(new int[]{1, 1}) == 2;
        assert Arrays.equals(productExceptSelf(new int[]{1, 2, 3, 4}), new int[]{24, 12, 8, 6});
        assert Arrays.equals(corpFlightBookings(new int[][]{{1, 2, 10}, {2, 3, 20}, {2, 5, 25}}, 5), new int[]{10, 55, 45, 25, 25});
        assert singleNumber(new int[]{4, 1, 2, 1, 2}) == 4;
        assert Arrays.equals(singleNumberIII(new int[]{1, 2, 1, 3, 2, 5}), new int[]{3, 5});
        assert findDuplicate(new int[]{1, 3, 4, 2, 2}) == 2;
        System.out.println("P01ArraysHashing OK");
    }
}
