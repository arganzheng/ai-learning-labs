import java.util.*;

/** 面试手撕代码（03）：栈、单调栈与单调队列。运行：java -ea P03StackMonotonic */
public class P03StackMonotonic {

    /** LC 20. 用 ArrayDeque 当栈（不要用 java.util.Stack）。 */
    static boolean isValid(String s) {
        Deque<Character> stack = new ArrayDeque<>();
        for (char c : s.toCharArray()) {
            if (c == '(') stack.push(')');
            else if (c == '[') stack.push(']');
            else if (c == '{') stack.push('}');
            else if (stack.isEmpty() || stack.pop() != c) return false;
        }
        return stack.isEmpty();
    }

    /** LC 150. 逆波兰求值；Java 的 / 本来就向零取整。 */
    static int evalRPN(String[] tokens) {
        Deque<Integer> stack = new ArrayDeque<>();
        for (String t : tokens) {
            switch (t) {
                case "+" -> stack.push(stack.pop() + stack.pop());
                case "*" -> stack.push(stack.pop() * stack.pop());
                case "-" -> { int b = stack.pop(), a = stack.pop(); stack.push(a - b); }
                case "/" -> { int b = stack.pop(), a = stack.pop(); stack.push(a / b); }
                default -> stack.push(Integer.parseInt(t));
            }
        }
        return stack.pop();
    }

    /** LC 394. 两个栈：重复次数栈 + 前缀串栈。 */
    static String decodeString(String s) {
        Deque<Integer> counts = new ArrayDeque<>();
        Deque<StringBuilder> prefixes = new ArrayDeque<>();
        StringBuilder cur = new StringBuilder();
        int num = 0;
        for (char c : s.toCharArray()) {
            if (Character.isDigit(c)) num = num * 10 + (c - '0');
            else if (c == '[') {
                counts.push(num); prefixes.push(cur);
                cur = new StringBuilder(); num = 0;
            } else if (c == ']') {
                StringBuilder prev = prefixes.pop();
                int k = counts.pop();
                for (int i = 0; i < k; i++) prev.append(cur);
                cur = prev;
            } else cur.append(c);
        }
        return cur.toString();
    }

    /** LC 739. 单调递减栈存下标。 */
    static int[] dailyTemperatures(int[] t) {
        int[] out = new int[t.length];
        Deque<Integer> stack = new ArrayDeque<>();
        for (int i = 0; i < t.length; i++) {
            while (!stack.isEmpty() && t[stack.peek()] < t[i]) {
                int j = stack.pop();
                out[j] = i - j;
            }
            stack.push(i);
        }
        return out;
    }

    /** LC 84. 单调递增栈 + 两端哨兵。 */
    static int largestRectangleArea(int[] heights) {
        int n = heights.length;
        int[] hs = new int[n + 2];
        System.arraycopy(heights, 0, hs, 1, n);
        Deque<Integer> stack = new ArrayDeque<>();
        stack.push(0);
        int best = 0;
        for (int i = 1; i < hs.length; i++) {
            while (hs[stack.peek()] > hs[i]) {
                int h = hs[stack.pop()];
                best = Math.max(best, h * (i - stack.peek() - 1));
            }
            stack.push(i);
        }
        return best;
    }

    /** LC 85. 每行柱状图。 */
    static int maximalRectangle(char[][] m) {
        if (m.length == 0) return 0;
        int[] heights = new int[m[0].length];
        int best = 0;
        for (char[] row : m) {
            for (int j = 0; j < row.length; j++) heights[j] = row[j] == '1' ? heights[j] + 1 : 0;
            best = Math.max(best, largestRectangleArea(heights));
        }
        return best;
    }

    /** LC 239. 单调递减双端队列。 */
    static int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length;
        int[] out = new int[n - k + 1];
        Deque<Integer> dq = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            while (!dq.isEmpty() && nums[dq.peekLast()] <= nums[i]) dq.pollLast();
            dq.addLast(i);
            if (dq.peekFirst() <= i - k) dq.pollFirst();
            if (i >= k - 1) out[i - k + 1] = nums[dq.peekFirst()];
        }
        return out;
    }

    /** LC 227. 栈存带符号的项。 */
    static int calculate(String s) {
        Deque<Integer> stack = new ArrayDeque<>();
        int num = 0;
        char op = '+';
        String t = s + "+";
        for (int i = 0; i < t.length(); i++) {
            char c = t.charAt(i);
            if (Character.isDigit(c)) num = num * 10 + (c - '0');
            else if (c != ' ') {
                switch (op) {
                    case '+' -> stack.push(num);
                    case '-' -> stack.push(-num);
                    case '*' -> stack.push(stack.pop() * num);
                    default -> stack.push(stack.pop() / num);
                }
                num = 0; op = c;
            }
        }
        int sum = 0;
        for (int v : stack) sum += v;
        return sum;
    }

    /** LC 155. 辅助栈存当前最小。 */
    static class MinStack {
        private final Deque<Integer> stack = new ArrayDeque<>(), mins = new ArrayDeque<>();
        void push(int v) { stack.push(v); mins.push(mins.isEmpty() ? v : Math.min(v, mins.peek())); }
        void pop() { stack.pop(); mins.pop(); }
        int top() { return stack.peek(); }
        int getMin() { return mins.peek(); }
    }

    public static void main(String[] args) {
        assert isValid("()[]{}") && !isValid("(]") && !isValid("(");
        assert evalRPN(new String[]{"4", "13", "5", "/", "+"}) == 6;
        assert decodeString("3[a2[c]]").equals("accaccacc");
        assert Arrays.equals(dailyTemperatures(new int[]{73, 74, 75, 71, 69, 72, 76, 73}), new int[]{1, 1, 4, 2, 1, 1, 0, 0});
        assert largestRectangleArea(new int[]{2, 1, 5, 6, 2, 3}) == 10;
        assert maximalRectangle(new char[][]{"10100".toCharArray(), "10111".toCharArray(), "11111".toCharArray(), "10010".toCharArray()}) == 6;
        assert Arrays.equals(maxSlidingWindow(new int[]{1, 3, -1, -3, 5, 3, 6, 7}, 3), new int[]{3, 3, 5, 5, 6, 7});
        assert calculate("3+2*2") == 7 && calculate(" 3+5 / 2 ") == 5 && calculate("14-3/2") == 13;
        MinStack ms = new MinStack();
        ms.push(-2); ms.push(0); ms.push(-3);
        assert ms.getMin() == -3;
        ms.pop();
        assert ms.top() == 0 && ms.getMin() == -2;
        System.out.println("P03StackMonotonic OK");
    }
}
