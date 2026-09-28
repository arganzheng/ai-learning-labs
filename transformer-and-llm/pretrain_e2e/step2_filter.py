"""第 2 步：过滤——语言识别 → Gopher 文档级质量规则 → Gopher 重复度规则 → C4 行级清理。
每条规则删掉多少、删掉的长什么样。输出 data/filtered.jsonl 与 out/e2e-2-filter-funnel.svg。

    python step2_filter.py
"""
import os
import re
import sys
from collections import Counter, OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from quality_filters import c4_lines, gopher_quality, gopher_repetition  # noqa: E402

from _plot import C, plt, save  # noqa: E402
from common import DATA, Timer, english_score, fmt, read_jsonl, write_jsonl  # noqa: E402

if __name__ == "__main__":
    docs = read_jsonl(f"{DATA}/raw.jsonl")
    n0 = len(docs)
    stages = OrderedDict([("原始", n0)])
    removed_examples = {}

    # 1. 语言：英文功能词比例
    with Timer("语言识别"):
        scored = [(english_score(d["text"]), d) for d in docs]
    en = [d for s, d in scored if s >= 0.12]
    non_en = [d for s, d in scored if s < 0.12]
    stages["英文（功能词比例 ≥ 0.12）"] = len(en)
    removed_examples["非英文"] = non_en[:3]
    agree = sum(d["lang_hint"].startswith("eng") for d in en) / max(1, len(en))
    print(f"语言识别：{fmt(n0)} → {fmt(len(en))} 篇英文（{len(en)/n0:.0%}）；与 Common Crawl 自己标的语言一致率 {agree:.0%}")

    # 2. Gopher 文档级质量
    with Timer("Gopher 质量规则"):
        fail_counter = Counter()
        kept, dropped = [], []
        for d in en:
            res = gopher_quality(d["text"])
            fails = [k for k, (ok, v) in res.items() if not ok]
            if fails:
                fail_counter[fails[0]] += 1
                d["_fail"] = fails[0]
                d["_val"] = res[fails[0]][1]
                dropped.append(d)
            else:
                kept.append(d)
    stages["Gopher 质量规则"] = len(kept)
    print(f"Gopher 质量规则：{fmt(len(en))} → {fmt(len(kept))}（删 {len(dropped)/len(en):.0%}）。每条规则首次击中的文档数：")
    for rule, n in fail_counter.most_common():
        ex = next(d for d in dropped if d["_fail"] == rule)
        print(f"   {rule:<22} {fmt(n):>7}   例：[{ex['_val']}] {ex['text'][:110].replace(chr(10), ' ⏎ ')!r}")
        removed_examples[rule] = [d for d in dropped if d["_fail"] == rule][:2]

    # 3. C4 行级：删掉不像句子的行（导航、页脚、按钮）
    with Timer("C4 行级规则"):
        before_chars = sum(len(d["text"]) for d in kept)
        final = []
        dropped_lines = Counter()
        for d in kept:
            keep, drop = c4_lines(d["text"])
            for l in drop:
                dropped_lines[l[:60]] += 1
            text = "\n".join(keep)
            if len(re.findall(r"\S+", text)) >= 50:
                final.append({"url": d["url"], "text": text})
        after_chars = sum(len(d["text"]) for d in final)
    stages["C4 行级清理后仍 ≥ 50 词"] = len(final)
    print(f"C4 行级规则：字符 {fmt(before_chars)} → {fmt(after_chars)}（删 {1-after_chars/before_chars:.0%} 的字符）；文档 {fmt(len(kept))} → {fmt(len(final))}")
    print("   被删得最多的行（跨文档重复出现，典型的导航 / 页脚）：")
    for l, n in dropped_lines.most_common(8):
        print(f"     {n:>5} ×  {l!r}")

    # 4. Gopher 重复度（放在行级清理之后：先把导航 / 页脚行删掉，再看正文本身重不重复）
    with Timer("Gopher 重复度规则"):
        kept2, dropped2 = [], []
        rep_counter = Counter()
        for d in final:
            res = gopher_repetition(d["text"])
            fails = [k for k, (ok, v) in res.items() if not ok]
            if fails:
                rep_counter[fails[0]] += 1
                d["_fail"] = fails[0]; d["_val"] = res[fails[0]][1]
                dropped2.append(d)
            else:
                kept2.append(d)
    stages["Gopher 重复度规则"] = len(kept2)
    print(f"Gopher 重复度规则：{fmt(len(final))} → {fmt(len(kept2))}（删 {len(dropped2)/len(final):.0%}）：")
    for rule, n in rep_counter.most_common(4):
        ex = next(d for d in dropped2 if d["_fail"] == rule)
        print(f"   {rule:<26} {fmt(n):>6}   例：[{ex['_val']}] {ex['text'][:110].replace(chr(10), ' ⏎ ')!r}")

    print(f"\n漏斗：" + " → ".join(f"{k} {fmt(v)}" for k, v in stages.items()))
    final = [{"url": d["url"], "text": d["text"]} for d in kept2]
    after_chars = sum(len(d["text"]) for d in final)
    print(f"最终 {fmt(len(final))} 篇，{fmt(after_chars)} 字符（{after_chars/1e6:.1f} MB），是原始文档数的 {len(final)/n0:.1%}")
    write_jsonl(final, f"{DATA}/filtered.jsonl")

    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    names = list(stages); vals = list(stages.values())
    bars = ax.barh(names[::-1], vals[::-1], color=[C["green"]] + [C["blue"]] * (len(vals) - 1))
    for b, v in zip(bars, vals[::-1]):
        ax.text(b.get_width() + n0 * 0.01, b.get_y() + b.get_height() / 2, f"{fmt(v)}（{v/n0:.0%}）", va="center", fontsize=8)
    ax.set(xlabel="文档数", title="过滤漏斗：两个 WET 文件里的 68,834 个网页，每步剩多少", xlim=(0, n0 * 1.25))
    save(fig, "e2e-2-filter-funnel")
