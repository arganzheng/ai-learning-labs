"""启发式质量过滤（Transformer 与 LLM 11）：实现 Gopher（Rae 等 2021）的文档级规则与重复度规则、
C4 的行级规则，对几段典型网页文本打印每条规则的判定。
https://arganzheng.life/pretraining-data-pipeline-dedup-filtering-and-mixture.html

纯 Python 标准库。  python quality_filters.py
"""
import re
from collections import Counter

STOP_WORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}


def words_of(text):
    return re.findall(r"\S+", text)


def gopher_quality(text):
    """Gopher 的文档级规则：返回 {规则: (通过?, 观测值)}。阈值取论文原值。"""
    words = words_of(text)
    n = len(words)
    lines = [l for l in text.split("\n") if l.strip()]
    alpha_words = sum(bool(re.search(r"[A-Za-z]", w)) for w in words)
    mean_len = sum(len(w) for w in words) / max(1, n)
    symbols = text.count("#") + text.count("...") + text.count("…")
    bullets = sum(l.lstrip().startswith(("-", "*", "•", "·")) for l in lines)
    ellipsis_end = sum(l.rstrip().endswith(("...", "…")) for l in lines)
    stop = sum(w.lower().strip(".,;:!?") in STOP_WORDS for w in words)
    return {
        "词数 50–100k": (50 <= n <= 100_000, n),
        "平均词长 3–10": (3 <= mean_len <= 10, round(mean_len, 1)),
        "#/… 与词数之比 < 0.1": (symbols / max(1, n) < 0.1, round(symbols / max(1, n), 3)),
        "以列表符开头的行 < 90%": (bullets / max(1, len(lines)) < 0.9, round(bullets / max(1, len(lines)), 2)),
        "以省略号结尾的行 < 30%": (ellipsis_end / max(1, len(lines)) < 0.3, round(ellipsis_end / max(1, len(lines)), 2)),
        "含字母的词 >= 80%": (alpha_words / max(1, n) >= 0.8, round(alpha_words / max(1, n), 2)),
        "停用词 >= 2 个": (stop >= 2, stop),
    }


def gopher_repetition(text):
    """Gopher 的重复度规则：重复行 / 段落的比例，高频 n-gram 占的字符比例。"""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    words = re.findall(r"\w+", text.lower())
    out = {}

    def dup_fraction(items):
        c = Counter(items)
        dup = sum(v for v in c.values() if v > 1)
        dup_chars = sum(len(k) * v for k, v in c.items() if v > 1)
        return dup / max(1, len(items)), dup_chars / max(1, sum(len(i) for i in items))

    f, fc = dup_fraction(lines)
    out["重复行比例 <= 0.30"] = (f <= 0.30, round(f, 2))
    out["重复行字符比例 <= 0.20"] = (fc <= 0.20, round(fc, 2))
    f, fc = dup_fraction(paras)
    out["重复段落比例 <= 0.30"] = (f <= 0.30, round(f, 2))
    total_chars = max(1, sum(len(w) for w in words))
    for n, thr in ((2, 0.20), (3, 0.18), (4, 0.16)):
        grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
        if grams:
            g, c = grams.most_common(1)[0]
            frac = c * sum(len(w) for w in g) / total_chars
        else:
            frac = 0.0
        out[f"最高频 {n}-gram 字符占比 <= {thr}"] = (frac <= thr, round(frac, 2))
    for n, thr in ((5, 0.15), (10, 0.10)):
        grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
        covered = set()                       # 被任何一个重复 n-gram 覆盖的词位置（不重复计数）
        for i in range(len(words) - n + 1):
            if grams[tuple(words[i : i + n])] > 1:
                covered.update(range(i, i + n))
        frac = sum(len(words[i]) for i in covered) / total_chars
        out[f"重复 {n}-gram 字符占比 <= {thr}"] = (frac <= thr, round(frac, 2))
    return out


def c4_lines(text):
    """C4 的行级规则：保留以终结标点结尾、至少 3 个词、不含 'lorem ipsum' / '{' / 'javascript' 提示的行。"""
    kept, dropped = [], []
    for l in text.split("\n"):
        s = l.strip()
        if not s:
            continue
        ok = s.endswith((".", "!", "?", '"', "”")) and len(s.split()) >= 3 \
            and "lorem ipsum" not in s.lower() and "{" not in s and "javascript" not in s.lower()
        (kept if ok else dropped).append(s)
    return kept, dropped


SAMPLES = {
    "正常文章": "Residual connections were introduced to make very deep networks trainable. Without them, the gradient "
              "that reaches the first layers is a product of many Jacobians, and any systematic deviation of their norms from one "
              "grows or shrinks exponentially with depth.\n"
              "With a residual path, each layer's Jacobian becomes the identity plus a perturbation, and the product keeps an "
              "identity component that carries gradient straight to the bottom of the network. This is the reason transformers "
              "with a hundred layers can be trained with the same optimizer as a network with two.",
    "导航栏 / 列表页": """Home
Products
- Laptops
- Phones
- Tablets
- Accessories
- Deals
Contact us
Sign in
© 2024 Example Corp. All rights reserved.""",
    "SEO 关键词堆砌": """cheap flights cheap flights cheap flights to paris cheap flights to paris cheap flights to london
best cheap flights cheap flights deals cheap flights cheap flights cheap flights to paris cheap flights to rome
cheap flights cheap flights cheap flights to paris cheap flights to paris cheap flights to london""",
    "表格 / 数字页": """2024-01-01  103.4  98.2  1,204  0.33 ...
2024-01-02  104.1  97.9  1,190  0.35 ...
2024-01-03  102.8  99.0  1,233  0.31 ...
2024-01-04  105.2  98.7  1,180  0.36 ...
2024-01-05  103.9  98.4  1,211  0.34 ...""",
    "带页眉页脚的正文": "Home | Blog | About\n"
                  "Attention lets every position in a sequence read from every other position, so information does not have to pass "
                  "through a chain of recurrent states. The cost is quadratic in sequence length, which is why the KV cache and the "
                  "attention variants that shrink it matter so much for long contexts.\n"
                  "Share this: Twitter Facebook LinkedIn\n"
                  "Copyright 2024. Privacy policy. Terms of service. Cookie settings {",
}


def main():
    for name, text in SAMPLES.items():
        q = gopher_quality(text)
        r = gopher_repetition(text)
        fails = [k for k, (ok, _) in {**q, **r}.items() if not ok]
        kept, dropped = c4_lines(text)
        verdict = "保留" if not fails else "丢弃"
        print(f"=== {name}：Gopher {verdict}（{len(fails)} 条规则未过）===")
        for k, (ok, v) in {**q, **r}.items():
            if not ok:
                print(f"    ✗ {k:<28} 观测 {v}")
        print(f"    C4 行级过滤：保留 {len(kept)} 行，丢弃 {len(dropped)} 行" + (f"，例如丢弃 {dropped[0][:40]!r}" if dropped else ""))
        print()
    print("规则的性质：全是可解释的阈值，几乎零成本，能清掉导航栏、关键词堆砌、表格与代码碎片这类明显垃圾；"
          "但分不出'正确的百科条目'与'流畅的胡说'——那是模型打分（FineWeb-Edu、DCLM 的分类器）要做的事。")


if __name__ == "__main__":
    main()
