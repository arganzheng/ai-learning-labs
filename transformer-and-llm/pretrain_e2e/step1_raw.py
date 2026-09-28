"""第 1 步：原料长什么样——读一个 Common Crawl WET 文件，看网页正文的真实样子。
输出 data/raw.jsonl（全部文档）与 out/e2e-1-raw.svg（长度分布 + 语言分布）。

    python step1_raw.py
"""
import os
import re
import sys
from collections import Counter

import numpy as np

from _plot import C, plt, save
from common import DATA, Timer, ensure_wets, fmt, read_wet, top_langs, write_jsonl

if __name__ == "__main__":
    paths = ensure_wets(partial_ok="--partial" in sys.argv)
    with Timer("读 WET"):
        docs = [d for p in paths for d in read_wet(p)]
    print(f"{len(paths)} 个 WET 文件：" + ", ".join(os.path.basename(p) for p in paths))
    chars = np.array([len(d["text"]) for d in docs])
    words = np.array([len(re.findall(r"\S+", d["text"])) for d in docs])
    print(f"文档数 {fmt(len(docs))}，共 {fmt(int(chars.sum()))} 字符（{chars.sum()/1e6:.0f} MB 文本）")
    print(f"每篇长度：中位数 {int(np.median(chars))} 字符，均值 {int(chars.mean())}，最长 {fmt(int(chars.max()))}；10% 的文档不到 {int(np.percentile(chars, 10))} 字符，10% 超过 {int(np.percentile(chars, 90))}")
    langs = top_langs(docs, 10)
    print("Common Crawl 自己标的语言（前 10）：", ", ".join(f"{l} {n} ({n/len(docs):.0%})" for l, n in langs))
    print("\n随机三篇（各取前 300 字符）：")
    rng = np.random.default_rng(0)
    for i in rng.choice(len(docs), 3, replace=False):
        d = docs[i]
        print(f"--- {d['url'][:90]}  [{d['lang_hint']}]  {len(d['text'])} 字符")
        print("   " + d["text"][:300].replace("\n", " ⏎ "))
    with Timer("写 raw.jsonl"):
        write_jsonl(docs, f"{DATA}/raw.jsonl")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8), width_ratios=[3, 2])
    axes[0].hist(np.log10(np.maximum(chars, 1)), bins=50, color=C["blue"])
    axes[0].set_xticks([1, 2, 3, 4, 5, 6], ["10", "100", "1K", "10K", "100K", "1M"])
    axes[0].set(xlabel="每篇字符数（对数刻度）", ylabel="文档数", title=f"{len(paths)} 个 WET 文件：{fmt(len(docs))} 个网页的正文长度")
    axes[0].axvline(np.log10(np.median(chars)), color=C["red"], ls="--", lw=1)
    axes[0].text(np.log10(np.median(chars)) + 0.1, axes[0].get_ylim()[1] * 0.9, f"中位数 {int(np.median(chars))}", color=C["red"], fontsize=8)
    names = [l for l, _ in langs[:7]] + ["其他"]
    vals = [n for _, n in langs[:7]] + [len(docs) - sum(n for _, n in langs[:7])]
    axes[1].barh(names[::-1], vals[::-1], color=[C["gray"]] + [C["blue"]] * 7)
    axes[1].set(title="Common Crawl 标的语言", xlabel="文档数")
    save(fig, "e2e-1-raw")
