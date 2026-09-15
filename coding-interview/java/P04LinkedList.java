import java.util.*;

/** 面试手撕代码（04）：链表。运行：java -ea P04LinkedList */
public class P04LinkedList {

    static class ListNode {
        int val;
        ListNode next;

        ListNode(int v) {
            val = v;
        }

        ListNode(int v, ListNode n) {
            val = v;
            next = n;
        }
    }

    static ListNode build(int... vals) {
        ListNode dummy = new ListNode(0), cur = dummy;
        for (int v : vals) {
            cur.next = new ListNode(v);
            cur = cur.next;
        }
        return dummy.next;
    }

    static List<Integer> toList(ListNode h) {
        List<Integer> out = new ArrayList<>();
        for (; h != null; h = h.next) out.add(h.val);
        return out;
    }

    /** LC 206. 三指针迭代。 */
    static ListNode reverseList(ListNode head) {
        ListNode prev = null, cur = head;
        while (cur != null) {
            ListNode nxt = cur.next;
            cur.next = prev;
            prev = cur;
            cur = nxt;
        }
        return prev;
    }

    /** LC 206 递归版。 */
    static ListNode reverseListRecursive(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode newHead = reverseListRecursive(head.next);
        head.next.next = head;
        head.next = null;
        return newHead;
    }

    /** LC 92. 哑节点 + 头插法。 */
    static ListNode reverseBetween(ListNode head, int left, int right) {
        ListNode dummy = new ListNode(0, head), pre = dummy;
        for (int i = 1; i < left; i++) pre = pre.next;
        ListNode cur = pre.next;
        for (int i = 0; i < right - left; i++) {
            ListNode nxt = cur.next;
            cur.next = nxt.next;
            nxt.next = pre.next;
            pre.next = nxt;
        }
        return dummy.next;
    }

    /** LC 25. K 个一组反转。 */
    static ListNode reverseKGroup(ListNode head, int k) {
        ListNode dummy = new ListNode(0, head), groupPrev = dummy;
        while (true) {
            ListNode kth = groupPrev;
            for (int i = 0; i < k && kth != null; i++) kth = kth.next;
            if (kth == null) return dummy.next;
            ListNode groupNext = kth.next, prev = groupNext, cur = groupPrev.next;
            while (cur != groupNext) {
                ListNode nxt = cur.next;
                cur.next = prev;
                prev = cur;
                cur = nxt;
            }
            ListNode tmp = groupPrev.next;
            groupPrev.next = kth;
            groupPrev = tmp;
        }
    }

    /** LC 876. */
    static ListNode middleNode(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }

    /** LC 142. 环入口。 */
    static ListNode detectCycle(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                ListNode p = head;
                while (p != slow) {
                    p = p.next;
                    slow = slow.next;
                }
                return p;
            }
        }
        return null;
    }

    /** LC 160. 两指针各走 A+B。 */
    static ListNode getIntersectionNode(ListNode a, ListNode b) {
        ListNode p = a, q = b;
        while (p != q) {
            p = p == null ? b : p.next;
            q = q == null ? a : q.next;
        }
        return p;
    }

    /** LC 21. */
    static ListNode mergeTwoLists(ListNode a, ListNode b) {
        ListNode dummy = new ListNode(0), tail = dummy;
        while (a != null && b != null) {
            if (a.val <= b.val) {
                tail.next = a;
                a = a.next;
            } else {
                tail.next = b;
                b = b.next;
            }
            tail = tail.next;
        }
        tail.next = a != null ? a : b;
        return dummy.next;
    }

    /** LC 23. PriorityQueue 做 k 路归并。O(N log k)。 */
    static ListNode mergeKLists(ListNode[] lists) {
        PriorityQueue<ListNode> pq = new PriorityQueue<>((x, y) -> Integer.compare(x.val, y.val));
        for (ListNode n : lists) if (n != null) pq.offer(n);
        ListNode dummy = new ListNode(0), tail = dummy;
        while (!pq.isEmpty()) {
            ListNode n = pq.poll();
            tail.next = n;
            tail = n;
            if (n.next != null) pq.offer(n.next);
        }
        return dummy.next;
    }

    /** LC 19. */
    static ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0, head), fast = dummy, slow = dummy;
        for (int i = 0; i <= n; i++) fast = fast.next;
        while (fast != null) {
            fast = fast.next;
            slow = slow.next;
        }
        slow.next = slow.next.next;
        return dummy.next;
    }

    /** LC 148. 归并排序。 */
    static ListNode sortList(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode slow = head, fast = head.next;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode mid = slow.next;
        slow.next = null;
        return mergeTwoLists(sortList(head), sortList(mid));
    }

    /** LC 234. 反转后半比较后恢复。 */
    static boolean isPalindrome(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode second = reverseList(slow);
        boolean ok = true;
        for (ListNode p = head, q = second; q != null; p = p.next, q = q.next)
            if (p.val != q.val) {
                ok = false;
                break;
            }
        reverseList(second);
        return ok;
    }

    static class RandomNode {
        int val;
        RandomNode next, random;

        RandomNode(int v) {
            val = v;
        }
    }

    /** LC 138. 哈希两趟。 */
    static RandomNode copyRandomList(RandomNode head) {
        Map<RandomNode, RandomNode> map = new HashMap<>();
        for (RandomNode c = head; c != null; c = c.next) map.put(c, new RandomNode(c.val));
        for (RandomNode c = head; c != null; c = c.next) {
            map.get(c).next = map.get(c.next);
            map.get(c).random = map.get(c.random);
        }
        return map.get(head);
    }

    public static void main(String[] args) {
        assert toList(reverseList(build(1, 2, 3, 4, 5))).equals(List.of(5, 4, 3, 2, 1));
        assert toList(reverseListRecursive(build(1, 2))).equals(List.of(2, 1));
        assert toList(reverseBetween(build(1, 2, 3, 4, 5), 2, 4)).equals(List.of(1, 4, 3, 2, 5));
        assert toList(reverseKGroup(build(1, 2, 3, 4, 5), 2)).equals(List.of(2, 1, 4, 3, 5));
        assert middleNode(build(1, 2, 3, 4, 5, 6)).val == 4;
        ListNode c = build(3, 2, 0, -4);
        c.next.next.next.next = c.next;
        assert detectCycle(c) == c.next;
        ListNode common = build(8, 4, 5), a = build(4, 1), b = build(5, 6, 1);
        a.next.next = common;
        b.next.next.next = common;
        assert getIntersectionNode(a, b) == common;
        assert toList(mergeKLists(new ListNode[] {build(1, 4, 5), build(1, 3, 4), build(2, 6)}))
                .equals(List.of(1, 1, 2, 3, 4, 4, 5, 6));
        assert toList(removeNthFromEnd(build(1), 1)).isEmpty();
        assert toList(sortList(build(-1, 5, 3, 4, 0))).equals(List.of(-1, 0, 3, 4, 5));
        ListNode p = build(1, 2, 2, 1);
        assert isPalindrome(p) && toList(p).equals(List.of(1, 2, 2, 1));
        RandomNode r1 = new RandomNode(7), r2 = new RandomNode(13);
        r1.next = r2;
        r2.random = r1;
        RandomNode cp = copyRandomList(r1);
        assert cp.next.random == cp && cp != r1;
        System.out.println("P04LinkedList OK");
    }
}
