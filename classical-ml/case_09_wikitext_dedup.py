"""案例（经典 ML 09）：给一个真实语料去重——wikitext-2 的 14,313 段 + 注入的 500 个近重复，MinHash-LSH vs 暴力。
https://arganzheng.life/deduplication-minhash-and-lsh-probabilities.html

    python case_09_wikitext_dedup.py

图输出到 out/case-09-*.svg。
"""
import hashlib
import time
from collections import defaultdict

import numpy as np

from _data import wikitext2
from _plot import C, plt, save

PRIME = (1 << 61) - 1


def shingles(text, n=5):
    """词级 5-gram 集合（预训练去重的常用设置）。"""
    w = text.lower().split()
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def jaccard(A, B):
    return len(A & B) / len(A | B)


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


class MinHash:
    def __init__(self, k=128, seed=0):
        r = np.random.default_rng(seed)
        self.a = r.integers(1, PRIME, k, dtype=np.int64)
        self.b = r.integers(0, PRIME, k, dtype=np.int64)

    def signature(self, S):
        x = np.array([h64(s) % PRIME for s in S], dtype=np.int64)[:, None]
        return ((x * self.a + self.b) % PRIME).min(0)


def perturb(text, rng, frac):
    """模拟近重复：随机替换 / 删掉 frac 比例的词（改写、模板变化、OCR 错字）。"""
    w = text.split()
    n = max(1, int(len(w) * frac))
    for j in rng.choice(len(w), n, replace=False):
        w[j] = "" if rng.random() < 0.5 else w[j][::-1]
    return " ".join(x for x in w if x)


def main():
    docs = wikitext2("train")
    n0 = len(docs)
    print(f"语料：wikitext-2 训练集里长度 > 200 字符的 {n0:,} 段，平均 {np.mean([len(d.split()) for d in docs]):.0f} 个词")

    # 注入 500 个近重复：改 2% / 5% / 10% / 20% 的词
    rng = np.random.default_rng(0)
    truth = []
    for i in range(500):
        src = int(rng.integers(0, n0)); frac = [0.02, 0.05, 0.10, 0.20][i % 4]
        docs.append(perturb(docs[src], rng, frac)); truth.append((src, n0 + i, frac))
    n = len(docs)
    print(f"注入 500 个近重复（改 2% / 5% / 10% / 20% 的词各 125 个）→ {n:,} 段")

    t = time.time()
    sets = [shingles(d) for d in docs]
    t_sh = time.time() - t
    print(f"\n=== 1. shingle：每段变成一个词级 5-gram 集合（{t_sh:.1f}s，平均 {np.mean([len(s) for s in sets]):.0f} 个 shingle）===")
    js = np.array([jaccard(sets[a], sets[b]) for a, b, _ in truth])
    for frac in [0.02, 0.05, 0.10, 0.20]:
        m = np.array([f == frac for _, _, f in truth])
        print(f"  改 {frac:.0%} 的词 → 与原文的 Jaccard 中位数 {np.median(js[m]):.2f}（词级 5-gram 很敏感：改一个词毁掉 5 个 shingle）")

    # 暴力：估算
    pairs_all = n * (n - 1) // 2
    t = time.time()
    k_try = 20000
    idx = rng.integers(0, n, (k_try, 2))
    _ = [jaccard(sets[a], sets[b]) for a, b in idx]
    per_pair = (time.time() - t) / k_try
    print("\n=== 2. 暴力两两比较 ===")
    print(f"  {n:,} 段有 {pairs_all:,} 对；实测每对 Jaccard {per_pair * 1e6:.0f} µs → 全部要 {pairs_all * per_pair / 60:.0f} 分钟")

    # MinHash + LSH
    b, r = 16, 8
    t = time.time()
    mh = MinHash(b * r)
    sigs = np.array([mh.signature(S) for S in sets])
    t_sig = time.time() - t
    t = time.time()
    buckets = defaultdict(list)
    for i, sig in enumerate(sigs):
        for band in range(b):
            buckets[(band, sig[band * r:(band + 1) * r].tobytes())].append(i)
    cand = set()
    for members in buckets.values():
        if len(members) > 1:
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    cand.add((min(members[x], members[y]), max(members[x], members[y])))
    t_lsh = time.time() - t
    t = time.time()
    verified = {p: jaccard(sets[p[0]], sets[p[1]]) for p in cand}
    t_ver = time.time() - t
    thr = 0.7
    dup = {p for p, j in verified.items() if j >= thr}
    print(f"\n=== 3. MinHash（{b * r} 个 hash）+ LSH（b={b} 段 × r={r}，阈值 ≈ (1/b)^(1/r) = {(1 / b) ** (1 / r):.2f}）===")
    print(f"  签名 {t_sig:.1f}s，分桶 + 收集候选 {t_lsh:.2f}s → {len(cand):,} 个候选对（全部对的 {len(cand) / pairs_all:.6%}）；对候选精确算 Jaccard {t_ver:.2f}s")
    print(f"  Jaccard ≥ {thr} 的 {len(dup):,} 对判为重复")
    truth_set = {(a, b) for a, b, _ in truth}
    found = truth_set & set(cand)
    print(f"\n  注入的 500 对里成为候选的 {len(found)}；按改动比例：")
    rec = {}
    for frac in [0.02, 0.05, 0.10, 0.20]:
        pf = [(a, b) for a, b, f in truth if f == frac]
        rec[frac] = np.mean([p in cand for p in pf]); jm = np.median([jaccard(sets[a], sets[b]) for a, b in pf])
        print(f"    改 {frac:>3.0%}：Jaccard 中位 {jm:.2f}，召回 {rec[frac]:.0%}")
    print("  解读：改 2% 的词（Jaccard 0.87）几乎全找到；改 5% 的 Jaccard 中位 0.68 正卡在阈值 0.71 上，召回 61%——S 曲线在阈值附近就是一半一半；"
          "改 10%（0.46）只剩 6%，改 20% 全漏。这不是 bug，是阈值的定义：词级 5-gram 下，改 10% 的词已经不算'近重复'。要抓更松的，调 b / r 或换更短的 shingle。")

    natural = sorted([(j, p) for p, j in verified.items() if j >= thr and p[0] < n0 and p[1] < n0], reverse=True)
    print(f"\n  语料自带的重复：候选里 Jaccard ≥ {thr} 且两段都是原文的有 {len(natural)} 对，例如：")
    for j, (a, bb) in natural[:3]:
        print(f"    J = {j:.2f}  段 {a}: {docs[a][:70]}...")
        print(f"             段 {bb}: {docs[bb][:70]}...")
    print("  同一系列条目（同型军舰、同一位作者的多篇）常有整段套模板改名字的写法，这是真实语料里去重会碰到的东西。")

    # 图：S 曲线 + 注入对的召回；耗时对比
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8), width_ratios=[1.4, 1])
    ax = axes[0]
    ss = np.linspace(0.2, 1, 200)
    ax.plot(ss, 1 - (1 - ss ** r) ** b, c=C["gray"], label=f"S 曲线 1 − (1 − s^{r})^{b}")
    for frac, col in zip([0.02, 0.05, 0.10, 0.20], [C["green"], C["blue"], C["orange"], C["red"]]):
        pts = [(jaccard(sets[a], sets[bb]), (a, bb) in cand) for a, bb, f in truth if f == frac]
        jj = np.array([x for x, _ in pts]); hit = np.array([h for _, h in pts])
        ax.scatter(jj, hit + rng.normal(0, 0.015, len(hit)), s=6, alpha=0.6, c=col, label=f"改 {frac:.0%} 的词（召回 {rec[frac]:.0%}）")
    ax.set_xlabel("与原文的真实 Jaccard"); ax.set_ylabel("成为候选（1）/ 漏掉（0）"); ax.legend(fontsize=6.5, loc="center left")
    ax.set_title("500 个注入的近重复：找到了哪些")
    ax = axes[1]
    ax.bar(["暴力\n（估算）", "MinHash\n签名", "LSH\n分桶", "候选\n精算"], [pairs_all * per_pair, t_sig, t_lsh, t_ver], color=[C["red"], C["blue"], C["blue"], C["blue"]])
    ax.set_yscale("log"); ax.set_ylabel("秒（对数刻度）"); ax.set_title(f"{n:,} 段：暴力 vs LSH")
    for i, v in enumerate([pairs_all * per_pair, t_sig, t_lsh, t_ver]):
        ax.text(i, v * 1.3, f"{v:.1f}s" if v < 100 else f"{v / 60:.0f} 分", ha="center", fontsize=7)
    save(fig, "case-09-lsh-wikitext")

    # 4. 判为重复的对怎么变成「删哪些段」：连通分量 vs 贪心，近重复不传递
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, bb in dup:
        ra, rb = find(a), find(bb)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    multi = sorted((g for g in groups.values() if len(g) > 1), key=len, reverse=True)
    sizes = np.array([len(g) for g in multi])
    below = 0
    inner = 0
    for g in multi:
        for x in range(len(g)):
            for y in range(x + 1, len(g)):
                inner += 1
                if verified.get((g[x], g[y]), 0.0) < thr:
                    below += 1
    kept_cc = n - int((sizes - 1).sum())
    print(f"\n=== 4. 从重复对到删除清单：{len(dup):,} 对 J ≥ {thr} 连成 {len(multi)} 个分量（最大 {sizes.max()} 段，大小 ≥ 3 的 {int((sizes >= 3).sum())} 个）===")
    note = "近重复关系不传递，分量把它们串在了一起" if below else "这份语料的重复几乎都是孤立的一对，没出现 A ≈ B ≈ C 而 A ≉ C 的链"
    print(f"  分量内两两 {inner} 对里有 {below} 对 J < {thr}（含不在候选里的对）——{note}")
    print(f"  策略 A 连通分量各留一段：保留 {kept_cc:,} 段、删 {n - kept_cc}")
    removed = set()
    for g in multi:
        rep = g[0]
        for i in g[1:]:
            if verified.get((rep, i), 0.0) >= thr:
                removed.add(i)
    same = "两种策略在这里删得一样多，分量小时策略之差不显现" if len(removed) == n - kept_cc else "两种策略对同一批重复对给出不同的删除数"
    print(f"  策略 B 只删与代表（分量里编号最小的一段）J ≥ {thr} 的：删 {len(removed)}，另外 {n - kept_cc - len(removed)} 段与代表不够像、被留下——{same}")
    biggest = multi[0]
    print("  最大分量的前 3 段：" + " / ".join(f"段 {i}: {docs[i][:40]}..." for i in biggest[:3]))


if __name__ == "__main__":
    main()
