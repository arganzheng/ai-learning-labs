import java.util.*;

/** 面试手撕代码（02）：双指针与滑动窗口。运行：java -ea P02TwoPointersSlidingWindow */
public class P02TwoPointersSlidingWindow {

    /** LC 3. 变长窗口，last[] 记录字符最后出现位置。O(n)。 */
    static int lengthOfLongestSubstring(String s) {
        int[] last = new int[128];
        Arrays.fill(last, -1);
        int left = 0, best = 0;
        for (int right = 0; right < s.length(); right++) {
            char ch = s.charAt(right);
            if (last[ch] >= left) left = last[ch] + 1;
            last[ch] = right;
            best = Math.max(best, right - left + 1);
        }
        return best;
    }

    /** LC 76. 最小覆盖子串：need 计数 + missing。O(|s| + |t|)。 */
    static String minWindow(String s, String t) {
        int[] need = new int[128];
        for (char c : t.toCharArray()) need[c]++;
        int missing = t.length(), left = 0, bestL = 0, bestR = Integer.MAX_VALUE;
        for (int right = 0; right < s.length(); right++) {
            if (need[s.charAt(right)]-- > 0) missing--;
            if (missing == 0) {
                while (need[s.charAt(left)] < 0) need[s.charAt(left++)]++;
                if (right - left < bestR - bestL) { bestL = left; bestR = right; }
                need[s.charAt(left++)]++;
                missing++;
            }
        }
        return bestR == Integer.MAX_VALUE ? "" : s.substring(bestL, bestR + 1);
    }

    /** LC 424. 窗口长 - 最高频 <= k；窗口只增不缩。O(n)。 */
    static int characterReplacement(String s, int k) {
        int[] count = new int[26];
        int left = 0, maxFreq = 0;
        for (int right = 0; right < s.length(); right++) {
            maxFreq = Math.max(maxFreq, ++count[s.charAt(right) - 'A']);
            if (right - left + 1 - maxFreq > k) count[s.charAt(left++) - 'A']--;
        }
        return s.length() - left;
    }

    /** LC 209. 和 >= target 的最短窗口。O(n)。 */
    static int minSubArrayLen(int target, int[] nums) {
        int left = 0, total = 0, best = Integer.MAX_VALUE;
        for (int right = 0; right < nums.length; right++) {
            total += nums[right];
            while (total >= target) {
                best = Math.min(best, right - left + 1);
                total -= nums[left++];
            }
        }
        return best == Integer.MAX_VALUE ? 0 : best;
    }

    /** LC 15. 排序 + 对撞双指针，三处去重。O(n^2)。 */
    static List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> out = new ArrayList<>();
        for (int i = 0; i < nums.length - 2; i++) {
            if (nums[i] > 0) break;
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            int lo = i + 1, hi = nums.length - 1;
            while (lo < hi) {
                int s = nums[i] + nums[lo] + nums[hi];
                if (s < 0) lo++;
                else if (s > 0) hi--;
                else {
                    out.add(Arrays.asList(nums[i], nums[lo], nums[hi]));
                    lo++; hi--;
                    while (lo < hi && nums[lo] == nums[lo - 1]) lo++;
                    while (lo < hi && nums[hi] == nums[hi + 1]) hi--;
                }
            }
        }
        return out;
    }

    /** LC 11. 对撞双指针：移动矮的一侧。 */
    static int maxArea(int[] h) {
        int lo = 0, hi = h.length - 1, best = 0;
        while (lo < hi) {
            best = Math.max(best, Math.min(h[lo], h[hi]) * (hi - lo));
            if (h[lo] < h[hi]) lo++; else hi--;
        }
        return best;
    }

    /** LC 42. 双指针接雨水。O(n) / O(1)。 */
    static int trap(int[] h) {
        int lo = 0, hi = h.length - 1, leftMax = 0, rightMax = 0, water = 0;
        while (lo < hi) {
            if (h[lo] < h[hi]) {
                leftMax = Math.max(leftMax, h[lo]);
                water += leftMax - h[lo++];
            } else {
                rightMax = Math.max(rightMax, h[hi]);
                water += rightMax - h[hi--];
            }
        }
        return water;
    }

    /** LC 567. 定长窗口 + 计数差。 */
    static boolean checkInclusion(String s1, String s2) {
        if (s1.length() > s2.length()) return false;
        int[] diff = new int[26];
        for (int i = 0; i < s1.length(); i++) { diff[s1.charAt(i) - 'a']++; diff[s2.charAt(i) - 'a']--; }
        int mismatch = 0;
        for (int d : diff) if (d != 0) mismatch++;
        if (mismatch == 0) return true;
        for (int i = s1.length(); i < s2.length(); i++) {
            int in = s2.charAt(i) - 'a', out = s2.charAt(i - s1.length()) - 'a';
            if (diff[in]-- == 0) mismatch++; else if (diff[in] == 0) mismatch--;
            if (diff[out]++ == 0) mismatch++; else if (diff[out] == 0) mismatch--;
            if (mismatch == 0) return true;
        }
        return false;
    }

    /** LC 283. 快慢指针原地分区。 */
    static void moveZeroes(int[] nums) {
        int slow = 0;
        for (int fast = 0; fast < nums.length; fast++) {
            if (nums[fast] != 0) {
                int t = nums[slow]; nums[slow] = nums[fast]; nums[fast] = t;
                slow++;
            }
        }
    }

    public static void main(String[] args) {
        assert lengthOfLongestSubstring("abcabcbb") == 3;
        assert lengthOfLongestSubstring("abba") == 2;
        assert minWindow("ADOBECODEBANC", "ABC").equals("BANC");
        assert minWindow("a", "aa").equals("");
        assert characterReplacement("AABABBA", 1) == 4;
        assert minSubArrayLen(7, new int[]{2, 3, 1, 2, 4, 3}) == 2;
        assert threeSum(new int[]{-1, 0, 1, 2, -1, -4}).equals(List.of(List.of(-1, -1, 2), List.of(-1, 0, 1)));
        assert maxArea(new int[]{1, 8, 6, 2, 5, 4, 8, 3, 7}) == 49;
        assert trap(new int[]{0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1}) == 6;
        assert checkInclusion("ab", "eidbaooo") && !checkInclusion("ab", "eidboaoo");
        int[] a = {0, 1, 0, 3, 12};
        moveZeroes(a);
        assert Arrays.equals(a, new int[]{1, 3, 12, 0, 0});
        System.out.println("P02TwoPointersSlidingWindow OK");
    }
}
