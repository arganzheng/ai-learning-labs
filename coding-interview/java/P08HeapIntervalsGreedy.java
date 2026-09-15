import java.util.*;

/** 面试手撕代码（08）：堆、Top-K、区间与贪心。运行：java -ea P08HeapIntervalsGreedy */
public class P08HeapIntervalsGreedy {

    /** LC 215. 大小 k 的最小堆。 */
    static int findKthLargestHeap(int[] nums, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for (int x : nums) {
            if (pq.size() < k) pq.offer(x);
            else if (x > pq.peek()) {
                pq.poll();
                pq.offer(x);
            }
        }
        return pq.peek();
    }

    /** LC 215. 快速选择（随机 pivot，三路分区）。 */
    static int findKthLargestQuickselect(int[] nums, int k) {
        Random rnd = new Random();
        int target = nums.length - k, lo = 0, hi = nums.length - 1;
        while (true) {
            int pivot = nums[lo + rnd.nextInt(hi - lo + 1)];
            int lt = lo, i = lo, gt = hi;
            while (i <= gt) {
                if (nums[i] < pivot) swap(nums, lt++, i++);
                else if (nums[i] > pivot) swap(nums, i, gt--);
                else i++;
            }
            if (target < lt) hi = lt - 1;
            else if (target > gt) lo = gt + 1;
            else return pivot;
        }
    }

    private static void swap(int[] a, int i, int j) {
        int t = a[i];
        a[i] = a[j];
        a[j] = t;
    }

    /** LC 347. 桶排序。 */
    static int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> count = new HashMap<>();
        for (int x : nums) count.merge(x, 1, Integer::sum);
        List<List<Integer>> buckets = new ArrayList<>();
        for (int i = 0; i <= nums.length; i++) buckets.add(new ArrayList<>());
        for (Map.Entry<Integer, Integer> e : count.entrySet())
            buckets.get(e.getValue()).add(e.getKey());
        int[] out = new int[k];
        int idx = 0;
        for (int c = nums.length; c > 0 && idx < k; c--)
            for (int x : buckets.get(c)) if (idx < k) out[idx++] = x;
        return out;
    }

    /** LC 295. 两个堆。 */
    static class MedianFinder {
        private final PriorityQueue<Integer> small =
                new PriorityQueue<>(Collections.reverseOrder());
        private final PriorityQueue<Integer> large = new PriorityQueue<>();

        void addNum(int num) {
            small.offer(num);
            large.offer(small.poll());
            if (large.size() > small.size()) small.offer(large.poll());
        }

        double findMedian() {
            return small.size() > large.size() ? small.peek() : (small.peek() + large.peek()) / 2.0;
        }
    }

    /** LC 56. */
    static int[][] mergeIntervals(int[][] intervals) {
        Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
        List<int[]> out = new ArrayList<>();
        for (int[] iv : intervals) {
            if (!out.isEmpty() && iv[0] <= out.get(out.size() - 1)[1])
                out.get(out.size() - 1)[1] = Math.max(out.get(out.size() - 1)[1], iv[1]);
            else out.add(new int[] {iv[0], iv[1]});
        }
        return out.toArray(new int[0][]);
    }

    /** LC 57. */
    static int[][] insertInterval(int[][] intervals, int[] nw) {
        List<int[]> out = new ArrayList<>();
        int i = 0, n = intervals.length, s = nw[0], e = nw[1];
        while (i < n && intervals[i][1] < s) out.add(intervals[i++]);
        while (i < n && intervals[i][0] <= e) {
            s = Math.min(s, intervals[i][0]);
            e = Math.max(e, intervals[i][1]);
            i++;
        }
        out.add(new int[] {s, e});
        while (i < n) out.add(intervals[i++]);
        return out.toArray(new int[0][]);
    }

    /** LC 435. 按右端点排序贪心。 */
    static int eraseOverlapIntervals(int[][] intervals) {
        Arrays.sort(intervals, (a, b) -> Integer.compare(a[1], b[1]));
        int kept = 0;
        long end = Long.MIN_VALUE;
        for (int[] iv : intervals)
            if (iv[0] >= end) {
                kept++;
                end = iv[1];
            }
        return intervals.length - kept;
    }

    /** LC 452. 注意坐标可到 ±2^31：比较用 Integer.compare 不要用减法。 */
    static int findMinArrowShots(int[][] points) {
        Arrays.sort(points, (a, b) -> Integer.compare(a[1], b[1]));
        int arrows = 0;
        long end = Long.MIN_VALUE;
        for (int[] p : points)
            if (p[0] > end) {
                arrows++;
                end = p[1];
            }
        return arrows;
    }

    /** LC 253. 最小堆存结束时间。 */
    static int minMeetingRooms(int[][] intervals) {
        Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
        PriorityQueue<Integer> ends = new PriorityQueue<>();
        for (int[] iv : intervals) {
            if (!ends.isEmpty() && ends.peek() <= iv[0]) ends.poll();
            ends.offer(iv[1]);
        }
        return ends.size();
    }

    /** LC 621. 数学公式。 */
    static int leastInterval(char[] tasks, int n) {
        int[] count = new int[26];
        int top = 0;
        for (char t : tasks) top = Math.max(top, ++count[t - 'A']);
        int ties = 0;
        for (int c : count) if (c == top) ties++;
        return Math.max(tasks.length, (top - 1) * (n + 1) + ties);
    }

    /** LC 55. */
    static boolean canJump(int[] nums) {
        int reach = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i > reach) return false;
            reach = Math.max(reach, i + nums[i]);
        }
        return true;
    }

    /** LC 45. */
    static int jump(int[] nums) {
        int steps = 0, end = 0, farthest = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            farthest = Math.max(farthest, i + nums[i]);
            if (i == end) {
                steps++;
                end = farthest;
            }
        }
        return steps;
    }

    /** LC 134. */
    static int canCompleteCircuit(int[] gas, int[] cost) {
        int total = 0, tank = 0, start = 0;
        for (int i = 0; i < gas.length; i++) {
            int d = gas[i] - cost[i];
            total += d;
            tank += d;
            if (tank < 0) {
                start = i + 1;
                tank = 0;
            }
        }
        return total >= 0 ? start : -1;
    }

    public static void main(String[] args) {
        assert findKthLargestHeap(new int[] {3, 2, 1, 5, 6, 4}, 2) == 5;
        assert findKthLargestQuickselect(new int[] {3, 2, 3, 1, 2, 4, 5, 5, 6}, 4) == 4;
        int[] tk = topKFrequent(new int[] {1, 1, 1, 2, 2, 3}, 2);
        Arrays.sort(tk);
        assert Arrays.equals(tk, new int[] {1, 2});
        MedianFinder mf = new MedianFinder();
        mf.addNum(1);
        mf.addNum(2);
        assert mf.findMedian() == 1.5;
        mf.addNum(3);
        assert mf.findMedian() == 2.0;
        assert Arrays.deepEquals(
                mergeIntervals(new int[][] {{1, 3}, {2, 6}, {8, 10}, {15, 18}}),
                new int[][] {{1, 6}, {8, 10}, {15, 18}});
        assert Arrays.deepEquals(
                insertInterval(
                        new int[][] {{1, 2}, {3, 5}, {6, 7}, {8, 10}, {12, 16}}, new int[] {4, 8}),
                new int[][] {{1, 2}, {3, 10}, {12, 16}});
        assert eraseOverlapIntervals(new int[][] {{1, 2}, {2, 3}, {3, 4}, {1, 3}}) == 1;
        assert findMinArrowShots(new int[][] {{10, 16}, {2, 8}, {1, 6}, {7, 12}}) == 2;
        assert findMinArrowShots(new int[][] {{-2147483646, -2147483645}, {2147483646, 2147483647}})
                == 2;
        assert minMeetingRooms(new int[][] {{0, 30}, {5, 10}, {15, 20}}) == 2;
        assert leastInterval("AAABBB".toCharArray(), 2) == 8;
        assert canJump(new int[] {2, 3, 1, 1, 4}) && !canJump(new int[] {3, 2, 1, 0, 4});
        assert jump(new int[] {2, 3, 1, 1, 4}) == 2;
        assert canCompleteCircuit(new int[] {1, 2, 3, 4, 5}, new int[] {3, 4, 5, 1, 2}) == 3;
        System.out.println("P08HeapIntervalsGreedy OK");
    }
}
