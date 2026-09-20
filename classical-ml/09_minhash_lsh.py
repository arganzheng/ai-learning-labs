"""去重（经典 ML 09）：Jaccard、MinHash 的无偏估计（含一个能手算的例子）、估计误差随 k 的变化、LSH 的 S 曲线、
在一批近重复文本上跑一遍、以及精确 / 模糊 / 语义三层去重的对照。
https://arganzheng.life/deduplication-minhash-and-lsh-probabilities.html

    python 09_minhash_lsh.py            # 全部：tiny estimate error scurve dedup semantic

图输出到 out/09-*.svg。
"""
import hashlib
import sys
from collections import defaultdict

import numpy as np

from _plot import C, plt, save

PRIME = (1 << 61) - 1


def shingles(text, n=5):
    """字符级 n-gram 集合（真实系统常用词级 5-gram）。"""
    text = " ".join(text.split())
    return {text[i:i + n] for i in range(max(1, len(text) - n + 1))}


def jaccard(A, B):
    return len(A & B) / len(A | B)


def h64(s):
    """确定性的 64 位 hash（Python 内建 hash 每次进程随机化，不能用）。"""
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


class MinHash:
    def __init__(self, k=128, seed=0):
        r = np.random.default_rng(seed)
        self.a = r.integers(1, PRIME, k, dtype=np.int64)
        self.b = r.integers(0, PRIME, k, dtype=np.int64)

    def signature(self, S):
        x = np.array([h64(s) % PRIME for s in S], dtype=np.int64)[:, None]        # [|S|, 1]
        hv = (x * self.a + self.b) % PRIME                                          # [|S|, k]：k 个随机 hash
        return hv.min(0)                                                            # 每个 hash 取最小值 → k 个签名


# ---------------- 0. 一个能手算的例子 ----------------
def exp_tiny():
    print("=== 0. 手算：两个小集合、三个随机排列 ===")
    U = ["a", "b", "c", "d", "e", "f"]
    A, B = {"a", "b", "c", "d"}, {"b", "c", "d", "e", "f"}
    print(f"  A = {sorted(A)}, B = {sorted(B)}；交 {sorted(A & B)}（{len(A & B)} 个），并 {sorted(A | B)}（{len(A | B)} 个）→ Jaccard = {len(A & B)}/{len(A | B)} = {jaccard(A, B):.3f}")
    perms = [["c", "a", "e", "b", "f", "d"], ["e", "d", "a", "c", "b", "f"], ["b", "f", "c", "a", "d", "e"]]
    eq = 0
    for i, p in enumerate(perms, 1):
        mA = min(A, key=p.index); mB = min(B, key=p.index)
        eq += mA == mB
        print(f"  排列 {i}: {' < '.join(p)}   A 里排最前的 {mA}，B 里排最前的 {mB} → {'相等' if mA == mB else '不等'}")
    print(f"  3 个签名里 {eq} 个相等 → 估计 Jaccard ≈ {eq}/3 = {eq/3:.3f}（真实 0.5；签名越多越准）")
    print("  为什么相等的概率恰好是 Jaccard：并集 6 个元素里谁排最前是等可能的；只有当它落在交集（3 个）里，A 与 B 的最小值才相同 → 3/6")
    print()


# ---------------- 1. MinHash 估计 Jaccard ----------------
def exp_estimate():
    print("=== 1. MinHash：签名相等的比例是 Jaccard 的无偏估计 ===")
    base = "the quick brown fox jumps over the lazy dog while the cat sleeps under the warm afternoon sun near the old wooden fence"
    words = base.split()
    r = np.random.default_rng(0)
    print(f"  {'改动词数':>6} {'真实 Jaccard':>12} {'k=16 估计':>10} {'k=128 估计':>11} {'k=1024 估计':>12}")
    for n_change in (0, 2, 5, 10, 20):
        w = words.copy()
        for i in r.choice(len(w), n_change, replace=False):
            w[i] = w[i][::-1]
        A, B = shingles(base), shingles(" ".join(w))
        est = [np.mean(MinHash(k).signature(A) == MinHash(k).signature(B)) for k in (16, 128, 1024)]
        print(f"  {n_change:>6} {jaccard(A, B):>12.3f} {est[0]:>10.3f} {est[1]:>11.3f} {est[2]:>12.3f}")
    print("  估计的标准差 ≈ √(J(1−J)/k)：k=128 时约 ±0.04；k 越大越准，代价是签名越长")
    print()


# ---------------- 1b. 估计误差随 k 的变化 ----------------
def exp_error():
    print("=== 1b. 估计的标准差 √(J(1−J)/k)：重复 200 次量出来 ===")
    r = np.random.default_rng(0)
    vocab = [f"w{i}" for i in range(2000)]
    base = " ".join(r.choice(vocab, 60))
    w = base.split()
    for j in r.choice(len(w), 12, replace=False): w[j] = r.choice(vocab)
    A, B = shingles(base), shingles(" ".join(w))
    J = jaccard(A, B)
    ks = [8, 16, 32, 64, 128, 256, 512, 1024]
    measured, theory = [], []
    for k in ks:
        est = np.array([np.mean(MinHash(k, seed=t).signature(A) == MinHash(k, seed=t).signature(B)) for t in range(200)])
        measured.append(est.std()); theory.append(np.sqrt(J * (1 - J) / k))
        print(f"  k = {k:>4}: 200 次估计的均值 {est.mean():.3f}（真实 {J:.3f}），标准差 实测 {est.std():.3f} / 理论 {theory[-1]:.3f}")
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    ax.plot(ks, measured, "o-", color=C["red"], label="实测（200 次重复）")
    ax.plot(ks, theory, "--", color=C["gray"], label="理论 √(J(1−J)/k)")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("签名个数 k"); ax.set_ylabel("Jaccard 估计的标准差"); ax.legend(frameon=False)
    ax.set_title(f"真实 Jaccard {J:.2f}：k 翻 4 倍，误差减半；k = 128 时 ±{theory[4]:.3f}", fontsize=8.5)
    save(fig, "09-minhash-error-vs-k")
    print()


# ---------------- 2. LSH 的 S 曲线 ----------------
def p_candidate(s, b, r):
    return 1 - (1 - s ** r) ** b


def exp_scurve():
    print("=== 2. LSH：k 个签名分 b 组、每组 r 个，任一组全等即候选；P = 1 − (1 − s^r)^b ===")
    print(f"  {'s':>5}", *[f"b={b},r={r}".rjust(12) for b, r in ((14, 8), (8, 16), (28, 4), (20, 5))])
    for s in (0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95):
        print(f"  {s:>5.2f}", *[f"{p_candidate(s, b, r):>12.4f}" for b, r in ((14, 8), (8, 16), (28, 4), (20, 5))])
    for b, r in ((14, 8), (8, 16), (28, 4)):
        s50 = (1 - 0.5 ** (1 / b)) ** (1 / r)
        print(f"  b={b:>2}, r={r:>2}: 曲线在 s ≈ {s50:.3f} 处过 50%   ← 这就是'阈值'；r 大更陡更严，b 大更宽松")
    print("  FineWeb 用 b=14, r=8（112 个 hash）：Jaccard 0.5 → 5% 候选，0.7 → 56%，0.8 → 92%，0.9 → 99.96%")
    print()


# ---------------- 3. 在一批文本上跑 MinHash + LSH ----------------
def exp_dedup():
    print("=== 3. 2000 段文本（含 300 组近重复）：暴力两两比较 vs LSH 分桶 ===")
    r = np.random.default_rng(1)
    vocab = [f"w{i}" for i in range(500)]
    docs = []
    for i in range(1700):
        docs.append(" ".join(r.choice(vocab, 40)))
    truth_pairs = []
    for i in range(300):                                   # 300 个近重复：改掉 1–6 个词
        src = r.integers(0, 1700)
        w = docs[src].split()
        for j in r.choice(len(w), r.integers(1, 7), replace=False):
            w[j] = r.choice(vocab)
        docs.append(" ".join(w))
        truth_pairs.append((src, 1700 + i))
    sets = [shingles(d, 5) for d in docs]
    b, rr = 14, 8
    mh = MinHash(b * rr)
    sigs = np.array([mh.signature(S) for S in sets])       # [2000, 112]
    buckets = defaultdict(list)
    for idx, sig in enumerate(sigs):
        for band in range(b):
            buckets[(band, sig[band * rr:(band + 1) * rr].tobytes())].append(idx)
    cand = set()
    for members in buckets.values():
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                cand.add((min(members[x], members[y]), max(members[x], members[y])))
    n_all = len(docs) * (len(docs) - 1) // 2
    hit = {p for p in cand if jaccard(sets[p[0]], sets[p[1]]) >= 0.7}
    found = set(truth_pairs) & hit
    js = np.array([jaccard(sets[a], sets[c]) for a, c in truth_pairs])
    print(f"  两两比较要算 {n_all:,} 对 Jaccard；LSH 只产生 {len(cand):,} 个候选对（{len(cand)/n_all*100:.2f}%），再对候选精确算")
    print(f"  300 组真实近重复的 Jaccard 分布: 最小 {js.min():.2f} / 中位 {np.median(js):.2f} / 最大 {js.max():.2f}")
    print(f"  Jaccard ≥ 0.7 的真实对 {int((js >= 0.7).sum())} 个，其中 LSH 找到 {len(found)}；候选里 Jaccard ≥ 0.7 的共 {len(hit)}（多出来的是碰巧相似的随机对）")
    print(f"  漏掉的都是 Jaccard 低于阈值的：改了 5–6 个词的对 J≈0.6，按 S 曲线只有约 {p_candidate(0.6, b, rr)*100:.0f}% 概率成候选——阈值就是这么定的")
    # 图：真实近重复对按 Jaccard 分桶，各桶被 LSH 找到的比例，叠上 S 曲线
    edges = np.linspace(0.6, 1.0, 9)
    frac, mids = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = [(p, j) for p, j in zip(truth_pairs, js) if lo <= j < hi]
        if not m: continue
        frac.append(np.mean([p in cand for p, _ in m])); mids.append((lo + hi) / 2)
    ss = np.linspace(0.3, 1, 200)
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    ax.plot(ss, p_candidate(ss, b, rr), color=C["gray"], label="S 曲线 1 − (1 − s⁸)¹⁴（理论）")
    ax.plot(mids, frac, "o", color=C["red"], label="300 组真实近重复：各 Jaccard 区间被找到的比例（实测）")
    ax.axvline(0.685, color=C["gray"], ls="--", lw=0.8)
    ax.set_xlabel("两段文本的真实 Jaccard"); ax.set_ylabel("成为候选的概率"); ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    save(fig, "09-lsh-recall-vs-jaccard")
    print()


# ---------------- 4. 三层去重：精确 / 模糊 / 语义 ----------------
def exp_semantic():
    print("=== 4. 三层去重：精确 hash、MinHash、embedding——对同一批句子各能抓到什么 ===")
    from _sentences import embeddings
    E, texts, labels, names = embeddings()
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    Ec = E - E.mean(0); Ec /= np.linalg.norm(Ec, axis=1, keepdims=True)          # 减均值（第八篇的各向异性修法）
    tmpl = [i for i, l in enumerate(labels) if names[l] == "模板"]
    sport = [i for i, l in enumerate(labels) if names[l] == "体育"]
    def pairs(idx):
        return [(a, b) for x, a in enumerate(idx) for b in idx[x + 1:]]
    for name, idx in (("6 句模板文本（换了数字）", tmpl), ("12 句体育（意思相关、措辞不同）", sport)):
        pj = [jaccard(shingles(texts[a], 5), shingles(texts[b], 5)) for a, b in pairs(idx)]
        pc = [Ec[a] @ Ec[b] for a, b in pairs(idx)]
        exact = sum(texts[a] == texts[b] for a, b in pairs(idx))
        print(f"  {name}：精确相同 {exact} 对；字符 5-gram Jaccard 均值 {np.mean(pj):.2f}（最小 {np.min(pj):.2f}）；减均值余弦 均值 {np.mean(pc):.2f}")
    cross = [Ec[a] @ Ec[b] for a in tmpl[:3] for b in sport[:3]]
    print(f"  模板 vs 体育 的减均值余弦均值 {np.mean(cross):.2f}（作对照）")
    print("  精确 hash 只抓逐字相同；MinHash 抓改了几个字的模板页（Jaccard 高）；意思相同但措辞不同的句子 Jaccard 低、只有 embedding 相似度能抓——语义去重（SemDeDup）就是在 embedding 上聚类再在簇内按余弦去重")
    print()


EXPS = {"tiny": exp_tiny, "estimate": exp_estimate, "error": exp_error, "scurve": exp_scurve, "dedup": exp_dedup, "semantic": exp_semantic}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    for n in names:
        EXPS[n]()
