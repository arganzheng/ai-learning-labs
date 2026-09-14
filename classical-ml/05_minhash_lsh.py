"""去重（经典 ML 05）：Jaccard、MinHash 的无偏估计、LSH 的 S 曲线、在一批近重复文本上跑一遍。纯 NumPy。
https://arganzheng.life/deduplication-minhash-and-lsh-probabilities.html

    python 05_minhash_lsh.py            # 全部：estimate scurve dedup
"""
import hashlib
import sys
from collections import defaultdict

import numpy as np

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
    truth_pairs = set()
    for i in range(300):                                   # 300 个近重复：改掉 1–6 个词
        src = r.integers(0, 1700)
        w = docs[src].split()
        for j in r.choice(len(w), r.integers(1, 7), replace=False):
            w[j] = r.choice(vocab)
        docs.append(" ".join(w))
        truth_pairs.add((src, 1700 + i))
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
    found = truth_pairs & hit
    js = np.array([jaccard(sets[a], sets[c]) for a, c in truth_pairs])
    print(f"  两两比较要算 {n_all:,} 对 Jaccard；LSH 只产生 {len(cand):,} 个候选对（{len(cand)/n_all*100:.2f}%），再对候选精确算")
    print(f"  300 组真实近重复的 Jaccard 分布: 最小 {js.min():.2f} / 中位 {np.median(js):.2f} / 最大 {js.max():.2f}")
    print(f"  Jaccard ≥ 0.7 的真实对 {int((js >= 0.7).sum())} 个，其中 LSH 找到 {len(found)}；候选里 Jaccard ≥ 0.7 的共 {len(hit)}（多出来的是碰巧相似的随机对）")
    print(f"  漏掉的都是 Jaccard 低于阈值的：改了 5–6 个词的对 J≈0.6，按 S 曲线只有约 {p_candidate(0.6, b, rr)*100:.0f}% 概率成候选——阈值就是这么定的")
    print()


EXPS = {"estimate": exp_estimate, "scurve": exp_scurve, "dedup": exp_dedup}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    for n in names:
        EXPS[n]()
