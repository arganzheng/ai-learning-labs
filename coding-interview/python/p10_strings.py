"""面试手撕代码（10）：字符串 —— 配套解法与测试。

https://arganzheng.life/coding-interview-strings.html
"""
from __future__ import annotations

import unittest
from collections import defaultdict
from functools import cmp_to_key
from typing import List


def longest_palindrome(s: str) -> str:
    """LC 5. 中心扩展：2n-1 个中心。O(n^2) / O(1)。"""
    best_l = best_r = 0

    def expand(l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1
        return l + 1, r - 1

    for i in range(len(s)):
        for l, r in (expand(i, i), expand(i, i + 1)):
            if r - l > best_r - best_l:
                best_l, best_r = l, r
    return s[best_l:best_r + 1]


def count_substrings(s: str) -> int:
    """LC 647. 同一中心扩展，改为计数。"""
    n = total = 0
    n = len(s)
    for center in range(2 * n - 1):
        l, r = center // 2, center // 2 + center % 2
        while l >= 0 and r < n and s[l] == s[r]:
            total += 1
            l -= 1
            r += 1
    return total


def manacher(s: str) -> str:
    """LC 5 的 O(n) 解（Manacher）：插分隔符统一奇偶，用镜像加速。"""
    t = "#" + "#".join(s) + "#"
    n = len(t)
    p = [0] * n                  # p[i]: 以 i 为中心的回文半径（不含中心）
    center = right = 0
    best_len = best_center = 0
    for i in range(n):
        if i < right:
            p[i] = min(right - i, p[2 * center - i])
        while i - p[i] - 1 >= 0 and i + p[i] + 1 < n and t[i - p[i] - 1] == t[i + p[i] + 1]:
            p[i] += 1
        if i + p[i] > right:
            center, right = i, i + p[i]
        if p[i] > best_len:
            best_len, best_center = p[i], i
    start = (best_center - best_len) // 2
    return s[start:start + best_len]


def build_lps(pattern: str) -> List[int]:
    """KMP 的失配表：lps[i] = pattern[:i+1] 的最长真前缀 = 真后缀 长度。"""
    lps = [0] * len(pattern)
    k = 0
    for i in range(1, len(pattern)):
        while k and pattern[i] != pattern[k]:
            k = lps[k - 1]
        if pattern[i] == pattern[k]:
            k += 1
        lps[i] = k
    return lps


def str_str(haystack: str, needle: str) -> int:
    """LC 28. KMP。O(n + m) / O(m)。"""
    if not needle:
        return 0
    lps = build_lps(needle)
    k = 0
    for i, ch in enumerate(haystack):
        while k and ch != needle[k]:
            k = lps[k - 1]
        if ch == needle[k]:
            k += 1
        if k == len(needle):
            return i - k + 1
    return -1


def group_anagrams(strs: List[str]) -> List[List[str]]:
    """LC 49. 26 位计数元组作 key（比排序 key 省一个 log）。O(NK)。"""
    groups = defaultdict(list)
    for s in strs:
        key = [0] * 26
        for ch in s:
            key[ord(ch) - 97] += 1
        groups[tuple(key)].append(s)
    return list(groups.values())


def my_atoi(s: str) -> int:
    """LC 8. 状态机：空白 → 符号 → 数字，溢出截断。"""
    INT_MAX, INT_MIN = 2 ** 31 - 1, -2 ** 31
    i, n = 0, len(s)
    while i < n and s[i] == " ":
        i += 1
    sign = 1
    if i < n and s[i] in "+-":
        sign = -1 if s[i] == "-" else 1
        i += 1
    num = 0
    while i < n and s[i].isdigit():
        num = num * 10 + (ord(s[i]) - 48)
        if sign * num > INT_MAX:            # Java / C++ 里要在乘 10 之前比较，见正文
            return INT_MAX
        if sign * num < INT_MIN:
            return INT_MIN
        i += 1
    return sign * num


def multiply(a: str, b: str) -> str:
    """LC 43. 竖式：a[i]*b[j] 落在 res[i+j+1]，最后统一进位。O(mn)。"""
    if a == "0" or b == "0":
        return "0"
    m, n = len(a), len(b)
    res = [0] * (m + n)
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            res[i + j + 1] += (ord(a[i]) - 48) * (ord(b[j]) - 48)
    for k in range(m + n - 1, 0, -1):
        res[k - 1] += res[k] // 10
        res[k] %= 10
    out = "".join(map(str, res)).lstrip("0")
    return out


def add_strings(a: str, b: str) -> str:
    """LC 415. 双指针从尾加，进位。"""
    i, j, carry = len(a) - 1, len(b) - 1, 0
    out: List[str] = []
    while i >= 0 or j >= 0 or carry:
        d = carry
        if i >= 0:
            d += ord(a[i]) - 48; i -= 1
        if j >= 0:
            d += ord(b[j]) - 48; j -= 1
        out.append(str(d % 10))
        carry = d // 10
    return "".join(reversed(out))


def largest_number(nums: List[int]) -> str:
    """LC 179. 自定义比较：a+b 与 b+a 谁大谁在前（该关系满足传递性）。"""
    strs = sorted(map(str, nums), key=cmp_to_key(lambda a, b: -1 if a + b > b + a else (1 if a + b < b + a else 0)))
    out = "".join(strs)
    return "0" if out[0] == "0" else out


def find_repeated_dna_sequences(s: str) -> List[str]:
    """LC 187. 滚动哈希（4 进制，20 位刚好 40 bit）。O(n)。"""
    if len(s) < 10:
        return []
    code = {"A": 0, "C": 1, "G": 2, "T": 3}
    mask = (1 << 20) - 1
    h = 0
    seen, out = set(), set()
    for i, ch in enumerate(s):
        h = ((h << 2) | code[ch]) & mask
        if i >= 9:
            if h in seen:
                out.add(s[i - 9:i + 1])
            seen.add(h)
    return sorted(out)


def reverse_words(s: str) -> str:
    """LC 151."""
    return " ".join(reversed(s.split()))


def longest_common_prefix(strs: List[str]) -> str:
    """LC 14. 纵向扫描。"""
    if not strs:
        return ""
    for i, ch in enumerate(strs[0]):
        for other in strs[1:]:
            if i >= len(other) or other[i] != ch:
                return strs[0][:i]
    return strs[0]


def compress(chars: List[str]) -> int:
    """LC 443. 读写双指针原地压缩。"""
    write = read = 0
    n = len(chars)
    while read < n:
        ch = chars[read]
        start = read
        while read < n and chars[read] == ch:
            read += 1
        chars[write] = ch
        write += 1
        if read - start > 1:
            for d in str(read - start):
                chars[write] = d
                write += 1
    return write


class Tests(unittest.TestCase):
    def test_palindromes(self):
        self.assertIn(longest_palindrome("babad"), ("bab", "aba"))
        self.assertEqual(longest_palindrome("cbbd"), "bb")
        self.assertEqual(manacher("babad"), "bab")
        self.assertEqual(manacher("cbbd"), "bb")
        self.assertEqual(manacher("abacdfgdcaba"), "aba")
        self.assertEqual(count_substrings("abc"), 3)
        self.assertEqual(count_substrings("aaa"), 6)

    def test_kmp(self):
        self.assertEqual(build_lps("aabaaab"), [0, 1, 0, 1, 2, 2, 3])
        self.assertEqual(str_str("sadbutsad", "sad"), 0)
        self.assertEqual(str_str("leetcode", "leeto"), -1)
        self.assertEqual(str_str("mississippi", "issip"), 4)
        self.assertEqual(str_str("aaaaab", "aab"), 3)

    def test_anagrams(self):
        out = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
        self.assertEqual(sorted(sorted(g) for g in out), [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]])

    def test_atoi(self):
        self.assertEqual(my_atoi("42"), 42)
        self.assertEqual(my_atoi("   -042"), -42)
        self.assertEqual(my_atoi("1337c0d3"), 1337)
        self.assertEqual(my_atoi("0-1"), 0)
        self.assertEqual(my_atoi("words and 987"), 0)
        self.assertEqual(my_atoi("-91283472332"), -2 ** 31)
        self.assertEqual(my_atoi("91283472332"), 2 ** 31 - 1)

    def test_big_numbers(self):
        self.assertEqual(multiply("123", "456"), "56088")
        self.assertEqual(multiply("2", "3"), "6")
        self.assertEqual(multiply("0", "52"), "0")
        self.assertEqual(add_strings("456", "77"), "533")
        self.assertEqual(add_strings("0", "0"), "0")

    def test_largest_number(self):
        self.assertEqual(largest_number([10, 2]), "210")
        self.assertEqual(largest_number([3, 30, 34, 5, 9]), "9534330")
        self.assertEqual(largest_number([0, 0]), "0")

    def test_misc(self):
        self.assertEqual(find_repeated_dna_sequences("AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"), ["AAAAACCCCC", "CCCCCAAAAA"])
        self.assertEqual(reverse_words("  hello world  "), "world hello")
        self.assertEqual(longest_common_prefix(["flower", "flow", "flight"]), "fl")
        chars = list("aabbccc")
        self.assertEqual(compress(chars), 6)
        self.assertEqual(chars[:6], list("a2b2c3"))


if __name__ == "__main__":
    unittest.main(verbosity=1)
