import java.util.*;

/** 面试手撕代码（10）：字符串。运行：java -ea P10Strings */
public class P10Strings {

    /** LC 5. 中心扩展。 */
    static String longestPalindrome(String s) {
        int bestL = 0, bestR = 0;
        for (int i = 0; i < s.length(); i++) {
            for (int[] lr : new int[][]{expand(s, i, i), expand(s, i, i + 1)}) {
                if (lr[1] - lr[0] > bestR - bestL) { bestL = lr[0]; bestR = lr[1]; }
            }
        }
        return s.substring(bestL, bestR + 1);
    }
    private static int[] expand(String s, int l, int r) {
        while (l >= 0 && r < s.length() && s.charAt(l) == s.charAt(r)) { l--; r++; }
        return new int[]{l + 1, r - 1};
    }

    /** LC 647. */
    static int countSubstrings(String s) {
        int n = s.length(), total = 0;
        for (int c = 0; c < 2 * n - 1; c++) {
            int l = c / 2, r = c / 2 + c % 2;
            while (l >= 0 && r < n && s.charAt(l) == s.charAt(r)) { total++; l--; r++; }
        }
        return total;
    }

    /** KMP 失配表。 */
    static int[] buildLps(String p) {
        int[] lps = new int[p.length()];
        for (int i = 1, k = 0; i < p.length(); i++) {
            while (k > 0 && p.charAt(i) != p.charAt(k)) k = lps[k - 1];
            if (p.charAt(i) == p.charAt(k)) k++;
            lps[i] = k;
        }
        return lps;
    }

    /** LC 28. KMP。 */
    static int strStr(String haystack, String needle) {
        if (needle.isEmpty()) return 0;
        int[] lps = buildLps(needle);
        for (int i = 0, k = 0; i < haystack.length(); i++) {
            while (k > 0 && haystack.charAt(i) != needle.charAt(k)) k = lps[k - 1];
            if (haystack.charAt(i) == needle.charAt(k)) k++;
            if (k == needle.length()) return i - k + 1;
        }
        return -1;
    }

    /** LC 49. 计数数组转字符串作 key。 */
    static List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> groups = new HashMap<>();
        for (String s : strs) {
            int[] cnt = new int[26];
            for (char c : s.toCharArray()) cnt[c - 'a']++;
            groups.computeIfAbsent(Arrays.toString(cnt), k -> new ArrayList<>()).add(s);
        }
        return new ArrayList<>(groups.values());
    }

    /** LC 8. 溢出必须在乘 10 之前判断。 */
    static int myAtoi(String s) {
        int i = 0, n = s.length(), sign = 1, num = 0;
        while (i < n && s.charAt(i) == ' ') i++;
        if (i < n && (s.charAt(i) == '+' || s.charAt(i) == '-')) sign = s.charAt(i++) == '-' ? -1 : 1;
        while (i < n && Character.isDigit(s.charAt(i))) {
            int d = s.charAt(i++) - '0';
            if (num > (Integer.MAX_VALUE - d) / 10) return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            num = num * 10 + d;
        }
        return sign * num;
    }

    /** LC 43. 竖式乘法。 */
    static String multiply(String a, String b) {
        if (a.equals("0") || b.equals("0")) return "0";
        int m = a.length(), n = b.length();
        int[] res = new int[m + n];
        for (int i = m - 1; i >= 0; i--)
            for (int j = n - 1; j >= 0; j--)
                res[i + j + 1] += (a.charAt(i) - '0') * (b.charAt(j) - '0');
        for (int k = m + n - 1; k > 0; k--) { res[k - 1] += res[k] / 10; res[k] %= 10; }
        StringBuilder sb = new StringBuilder();
        for (int d : res) if (!(sb.length() == 0 && d == 0)) sb.append(d);
        return sb.toString();
    }

    /** LC 415. */
    static String addStrings(String a, String b) {
        StringBuilder sb = new StringBuilder();
        for (int i = a.length() - 1, j = b.length() - 1, carry = 0; i >= 0 || j >= 0 || carry > 0; i--, j--) {
            int d = carry + (i >= 0 ? a.charAt(i) - '0' : 0) + (j >= 0 ? b.charAt(j) - '0' : 0);
            sb.append(d % 10);
            carry = d / 10;
        }
        return sb.reverse().toString();
    }

    /** LC 179. 自定义比较器。 */
    static String largestNumber(int[] nums) {
        String[] strs = Arrays.stream(nums).mapToObj(String::valueOf).toArray(String[]::new);
        Arrays.sort(strs, (x, y) -> (y + x).compareTo(x + y));
        if (strs[0].equals("0")) return "0";
        return String.join("", strs);
    }

    /** LC 187. 滚动哈希（2 bit 一个碱基）。 */
    static List<String> findRepeatedDnaSequences(String s) {
        if (s.length() < 10) return List.of();
        Map<Character, Integer> code = Map.of('A', 0, 'C', 1, 'G', 2, 'T', 3);
        int mask = (1 << 20) - 1, h = 0;
        Set<Integer> seen = new HashSet<>();
        TreeSet<String> out = new TreeSet<>();
        for (int i = 0; i < s.length(); i++) {
            h = ((h << 2) | code.get(s.charAt(i))) & mask;
            if (i >= 9) {
                if (!seen.add(h)) out.add(s.substring(i - 9, i + 1));
            }
        }
        return new ArrayList<>(out);
    }

    /** LC 151. */
    static String reverseWords(String s) {
        String[] parts = s.trim().split("\\s+");
        Collections.reverse(Arrays.asList(parts));
        return String.join(" ", parts);
    }

    public static void main(String[] args) {
        assert longestPalindrome("cbbd").equals("bb");
        assert countSubstrings("aaa") == 6;
        assert Arrays.equals(buildLps("aabaaab"), new int[]{0, 1, 0, 1, 2, 2, 3});
        assert strStr("mississippi", "issip") == 4 && strStr("leetcode", "leeto") == -1;
        assert groupAnagrams(new String[]{"eat", "tea", "tan", "ate", "nat", "bat"}).size() == 3;
        assert myAtoi("   -042") == -42 && myAtoi("1337c0d3") == 1337 && myAtoi("words 987") == 0;
        assert myAtoi("-91283472332") == Integer.MIN_VALUE && myAtoi("91283472332") == Integer.MAX_VALUE;
        assert multiply("123", "456").equals("56088") && multiply("0", "52").equals("0");
        assert addStrings("456", "77").equals("533");
        assert largestNumber(new int[]{3, 30, 34, 5, 9}).equals("9534330") && largestNumber(new int[]{0, 0}).equals("0");
        assert findRepeatedDnaSequences("AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT").equals(List.of("AAAAACCCCC", "CCCCCAAAAA"));
        assert reverseWords("  hello world  ").equals("world hello");
        System.out.println("P10Strings OK");
    }
}
