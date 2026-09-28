"""第 3 步：去重——精确去重（整篇哈希）+ MinHash/LSH 近重复去重（numpy 向量化）。
输出 data/clean.jsonl 与 out/e2e-3-dedup.svg。

    python step3_dedup.py
"""
import hashlib
import re
from collections import defaultdict

import numpy as np

from _plot import C, plt, save
from common import DATA, Timer, fmt, read_jsonl, write_jsonl

N_HASH, BANDS = 128, 16          # 16 段 × 8 行：Jaccard 0.7 的对约 62% 成为候选，0.9 的 99.9%
ROWS = N_HASH // BANDS
P = (1 << 61) - 1


def shingle_ids(text, n=5):
    words = re.findall(r"\w+", text.lower())
    grams = {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}
    return np.array([int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big") for g in grams], dtype=np.uint64)


if __name__ == "__main__":
    docs = read_jsonl(f"{DATA}/filtered.jsonl")
    n0 = len(docs)

    with Timer("精确去重"):
        seen, exact_kept, exact_dups = {}, [], []
        for d in docs:
            h = hashlib.sha1(d["text"].encode()).hexdigest()
            if h in seen:
                exact_dups.append((seen[h], d))
            else:
                seen[h] = d; exact_kept.append(d)
    print(f"精确去重：{fmt(n0)} → {fmt(len(exact_kept))}，删掉 {len(exact_dups)} 篇逐字相同的文档")
    for a, b in exact_dups[:2]:
        print(f"   例：{a['url'][:70]}  ==  {b['url'][:70]}")

    with Timer("MinHash 签名"):
        rng = np.random.default_rng(0)
        A = rng.integers(1, P, N_HASH, dtype=np.uint64); B = rng.integers(0, P, N_HASH, dtype=np.uint64)
        sigs = np.empty((len(exact_kept), N_HASH), dtype=np.uint64)
        shingle_sets = []
        for i, d in enumerate(exact_kept):
            sh = shingle_ids(d["text"])
            shingle_sets.append(sh)
            # 每个哈希函数 h(x) = a·x + b（uint64 环绕，相当于 mod 2^64），对整篇的 shingle 取最小值
            hv = sh[None, :] * A[:, None] + B[:, None]
            sigs[i] = hv.min(axis=1)
    print(f"签名：{fmt(len(exact_kept))} 篇 × {N_HASH} 个哈希；平均每篇 {np.mean([len(s) for s in shingle_sets]):.0f} 个 5-gram")

    with Timer("LSH 分桶 + 候选对验证"):
        buckets = defaultdict(list)
        for i in range(len(exact_kept)):
            for b in range(BANDS):
                buckets[(b, sigs[i, b * ROWS:(b + 1) * ROWS].tobytes())].append(i)
        cands = set()
        for ids in buckets.values():
            if 1 < len(ids) <= 50:
                for x in range(len(ids)):
                    for y in range(x + 1, len(ids)):
                        cands.add((ids[x], ids[y]))
        near = []
        for i, j in cands:
            a, b = shingle_sets[i], shingle_sets[j]
            jac = len(np.intersect1d(a, b)) / max(1, len(np.union1d(a, b)))
            if jac >= 0.7:
                near.append((i, j, jac))
    print(f"LSH：{fmt(len(cands))} 个候选对（暴力要比 {fmt(len(exact_kept)*(len(exact_kept)-1)//2)} 对），其中 Jaccard ≥ 0.7 的近重复 {fmt(len(near))} 对")
    drop = set()
    for i, j, jac in near:
        drop.add(max(i, j))
    clean = [d for k, d in enumerate(exact_kept) if k not in drop]
    print(f"近重复去重：{fmt(len(exact_kept))} → {fmt(len(clean))}（删 {len(drop)} 篇）")
    near.sort(key=lambda t: -t[2])
    for i, j, jac in near[:3]:
        a, b = exact_kept[i], exact_kept[j]
        print(f"   例 Jaccard {jac:.2f}：\n     {a['url'][:80]}\n     {b['url'][:80]}\n     A: {a['text'][:120]!r}\n     B: {b['text'][:120]!r}")
    print(f"\n最终语料：{fmt(len(clean))} 篇，{fmt(sum(len(d['text']) for d in clean))} 字符")
    write_jsonl(clean, f"{DATA}/clean.jsonl")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    jacs = [jac for _, _, jac in near]
    if jacs:
        axes[0].hist(jacs, bins=np.linspace(0.7, 1.0, 16), color=C["blue"])
    axes[0].set(xlabel="近重复对的 Jaccard 相似度", ylabel="对数", title=f"LSH 找到的 {len(near)} 对近重复")
    js = np.linspace(0, 1, 200)
    axes[1].plot(js, 1 - (1 - js ** ROWS) ** BANDS, color=C["red"])
    axes[1].axvline(0.7, color=C["gray"], ls=":", lw=1)
    axes[1].set(xlabel="两篇文档的真实 Jaccard", ylabel="成为候选对的概率", title=f"LSH 的 S 曲线（{BANDS} 段 × {ROWS} 行）")
    save(fig, "e2e-3-dedup")
