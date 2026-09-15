import java.util.*;
import java.util.function.IntPredicate;

/** 面试手撕代码（07）：二分。运行：java -ea P07BinarySearch */
public class P07BinarySearch {

    /** 在 [lo, hi) 上返回第一个 pred 为真的位置；全假返回 hi。 */
    static int firstTrue(int lo, int hi, IntPredicate pred) {
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;          // 不写 (lo+hi)/2：Java int 会溢出
            if (pred.test(mid)) hi = mid; else lo = mid + 1;
        }
        return lo;
    }

    static int lowerBound(int[] a, int x) { return firstTrue(0, a.length, i -> a[i] >= x); }
    static int upperBound(int[] a, int x) { return firstTrue(0, a.length, i -> a[i] > x); }

    /** LC 34. */
    static int[] searchRange(int[] nums, int target) {
        int lo = lowerBound(nums, target);
        if (lo == nums.length || nums[lo] != target) return new int[]{-1, -1};
        return new int[]{lo, upperBound(nums, target) - 1};
    }

    /** LC 33. */
    static int searchRotated(int[] nums, int target) {
        int lo = 0, hi = nums.length - 1;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (nums[mid] == target) return mid;
            if (nums[lo] <= nums[mid]) {
                if (nums[lo] <= target && target < nums[mid]) hi = mid - 1; else lo = mid + 1;
            } else {
                if (nums[mid] < target && target <= nums[hi]) lo = mid + 1; else hi = mid - 1;
            }
        }
        return -1;
    }

    /** LC 81. */
    static boolean searchRotatedII(int[] nums, int target) {
        int lo = 0, hi = nums.length - 1;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (nums[mid] == target) return true;
            if (nums[lo] == nums[mid] && nums[mid] == nums[hi]) { lo++; hi--; }
            else if (nums[lo] <= nums[mid]) {
                if (nums[lo] <= target && target < nums[mid]) hi = mid - 1; else lo = mid + 1;
            } else {
                if (nums[mid] < target && target <= nums[hi]) lo = mid + 1; else hi = mid - 1;
            }
        }
        return false;
    }

    /** LC 153. */
    static int findMinRotated(int[] nums) {
        int last = nums[nums.length - 1];
        return nums[firstTrue(0, nums.length - 1, m -> nums[m] <= last)];
    }

    /** LC 162. */
    static int findPeakElement(int[] nums) { return firstTrue(0, nums.length - 1, m -> nums[m] > nums[m + 1]); }

    /** LC 69. 用 long 防 m*m 溢出。 */
    static int mySqrt(int x) { return firstTrue(0, x + 1, m -> (long) m * m > x) - 1; }

    /** LC 875. 答案二分。 */
    static int minEatingSpeed(int[] piles, int h) {
        int max = Arrays.stream(piles).max().getAsInt();
        return firstTrue(1, max + 1, k -> {
            long hours = 0;
            for (int p : piles) hours += (p + k - 1) / k;
            return hours <= h;
        });
    }

    /** LC 1011（与 LC 410 同构）. */
    static int shipWithinDays(int[] weights, int days) {
        int max = 0, sum = 0;
        for (int w : weights) { max = Math.max(max, w); sum += w; }
        return firstTrue(max, sum + 1, cap -> {
            int used = 1, cur = 0;
            for (int w : weights) { if (cur + w > cap) { used++; cur = 0; } cur += w; }
            return used <= days;
        });
    }

    static int splitArray(int[] nums, int k) { return shipWithinDays(nums, k); }

    /** LC 378. 值域二分，阶梯计数。 */
    static int kthSmallestMatrix(int[][] matrix, int k) {
        int n = matrix.length;
        return firstTrue(matrix[0][0], matrix[n - 1][n - 1] + 1, x -> {
            int i = n - 1, j = 0, c = 0;
            while (i >= 0 && j < n) { if (matrix[i][j] <= x) { c += i + 1; j++; } else i--; }
            return c >= k;
        });
    }

    /** LC 4. 在短数组上二分切分点。 */
    static double findMedianSortedArrays(int[] a, int[] b) {
        if (a.length > b.length) return findMedianSortedArrays(b, a);
        int m = a.length, n = b.length, half = (m + n + 1) / 2, lo = 0, hi = m;
        while (lo <= hi) {
            int i = lo + (hi - lo) / 2, j = half - i;
            int aLeft = i > 0 ? a[i - 1] : Integer.MIN_VALUE, aRight = i < m ? a[i] : Integer.MAX_VALUE;
            int bLeft = j > 0 ? b[j - 1] : Integer.MIN_VALUE, bRight = j < n ? b[j] : Integer.MAX_VALUE;
            if (aLeft <= bRight && bLeft <= aRight) {
                if ((m + n) % 2 == 1) return Math.max(aLeft, bLeft);
                return (Math.max(aLeft, bLeft) + (double) Math.min(aRight, bRight)) / 2;
            }
            if (aLeft > bRight) hi = i - 1; else lo = i + 1;
        }
        throw new IllegalArgumentException("inputs not sorted");
    }

    public static void main(String[] args) {
        int[] a = {1, 2, 2, 2, 5, 7};
        assert lowerBound(a, 2) == 1 && upperBound(a, 2) == 4 && lowerBound(a, 8) == 6;
        assert Arrays.equals(searchRange(new int[]{5, 7, 7, 8, 8, 10}, 8), new int[]{3, 4});
        assert searchRotated(new int[]{4, 5, 6, 7, 0, 1, 2}, 0) == 4 && searchRotated(new int[]{1}, 0) == -1;
        assert searchRotatedII(new int[]{1, 0, 1, 1, 1}, 0) && !searchRotatedII(new int[]{2, 5, 6, 0, 0, 1, 2}, 3);
        assert findMinRotated(new int[]{4, 5, 6, 7, 0, 1, 2}) == 0 && findMinRotated(new int[]{11, 13}) == 11;
        assert findPeakElement(new int[]{1, 2, 3, 1}) == 2;
        assert mySqrt(2147395599) == 46339 && mySqrt(0) == 0;
        assert minEatingSpeed(new int[]{3, 6, 7, 11}, 8) == 4;
        assert shipWithinDays(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 5) == 15;
        assert splitArray(new int[]{7, 2, 5, 10, 8}, 2) == 18;
        assert kthSmallestMatrix(new int[][]{{1, 5, 9}, {10, 11, 13}, {12, 13, 15}}, 8) == 13;
        assert findMedianSortedArrays(new int[]{1, 3}, new int[]{2}) == 2.0;
        assert findMedianSortedArrays(new int[]{1, 2}, new int[]{3, 4}) == 2.5;
        assert findMedianSortedArrays(new int[]{}, new int[]{1}) == 1.0;
        System.out.println("P07BinarySearch OK");
    }
}
