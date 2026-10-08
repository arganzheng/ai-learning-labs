"""去重（经典 ML 09）：Jaccard、MinHash 的无偏估计（含一个能手算的例子）、估计误差随 k 的变化、LSH 的 S 曲线、
在一批近重复文本上跑一遍、以及精确 / 模糊 / 语义三层去重的对照。
https://arganzheng.life/deduplication-minhash-and-lsh-probabilities.html

    python 09_minhash_lsh.py            # 全部：tiny estimate error scurve dedup semantic

图输出到 out/09-*.svg。
"""
import hashlib
import sys
from collections import defaultdict
from itertools import pairwise

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
    for lo, hi in pairwise(edges):
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
    Ec = E - E.mean(0); Ec /= np.linalg.norm(Ec, axis=1, keepdims=True)          # 减均值（第八篇的各向异性修法）
    tmpl = [i for i, label in enumerate(labels) if names[label] == "模板"]
    sport = [i for i, label in enumerate(labels) if names[label] == "体育"]
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


# ---------------- 5. 可跟踪的小例子：shingle → 签名 → 分 band → 候选 → 精确复核 → 保留代表 ----------------
TRACE_DOCS = [
    "the cat sat on the mat by the door",
    "the cat sat on the mat by the window",
    "a cat sat on the mat by the window today",
    "machine learning models need lots of data",
    "machine learning models need lots of good data",
    "the quick brown fox jumps over the lazy dog",
]


def word_shingles(text, n=2):
    """词级 n-gram 集合：小例子用 2-gram，让 shingle 数少到能逐个看。"""
    w = text.split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def perm_signature(S, perms, universe):
    """独立随机排列下的 MinHash：签名 = 把全集打乱后，集合里排得最前的那个位置（理论公式的前提）。"""
    cols = [universe[s] for s in S]
    return perms[:, cols].min(1)


def band_keys(sig, b, r):
    return [tuple(int(v) for v in sig[band * r:(band + 1) * r]) for band in range(b)]


def lsh_candidates(sigs, b, r):
    """返回 {(i, j): [命中的 band 编号]}：任一 band 的 r 个签名全等即成候选。"""
    buckets = defaultdict(list)
    for i, sig in enumerate(sigs):
        for band, key in enumerate(band_keys(sig, b, r)):
            buckets[(band, key)].append(i)
    cand = defaultdict(list)
    for (band, _), members in buckets.items():
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                cand[(members[x], members[y])].append(band)
    return dict(cand)


def exact_pairs(sets, thr=0.0):
    out = {}
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            J = jaccard(sets[i], sets[j])
            if J > thr:
                out[(i, j)] = J
    return out


def exp_trace():
    print("=== 5. 可跟踪的小例子：6 段文本、词级 2-gram、k = 6 个随机排列分 b = 3 组 × r = 2，阈值 0.7 ===")
    k, b, r, thr, seed = 6, 3, 2, 0.7, 9
    sets = [word_shingles(d) for d in TRACE_DOCS]
    universe = {s: i for i, s in enumerate(sorted(set().union(*sets)))}
    for i, (d, S) in enumerate(zip(TRACE_DOCS, sets)):
        print(f"  d{i}: \"{d}\" → {len(S)} 个 shingle")
    print(f"  全集 {len(universe)} 个不同 shingle；d0 ∩ d1 = {sorted(sets[0] & sets[1])}")
    print(f"  d0 ∪ d1 多出来的：{sorted(sets[0] ^ sets[1])}")
    exact = exact_pairs(sets)
    print("  精确 Jaccard（只列有交集的对）：" + "，".join(f"(d{i},d{j}) {J:.3f}" for (i, j), J in sorted(exact.items())))
    rng = np.random.default_rng(seed)
    perms = np.array([rng.permutation(len(universe)) for _ in range(k)])     # [k, |U|]：k 个独立随机排列
    sigs = np.array([perm_signature(S, perms, universe) for S in sets])    # [6, k]
    print(f"  签名矩阵（seed={seed}；每列一个排列，值 = 排列后集合里最小的位置；竖线分 band）：")
    for i, sig in enumerate(sigs):
        cells = " | ".join(" ".join(f"{v:2d}" for v in key) for key in band_keys(sig, b, r))
        print(f"    d{i}: {cells}")
    cand = lsh_candidates(sigs, b, r)
    print("  候选对（某个 band 的 2 个签名全等）与精确复核：")
    for (i, j), bands in sorted(cand.items()):
        J = jaccard(sets[i], sets[j])
        eq = int((sigs[i] == sigs[j]).sum())
        verdict = f"J ≥ {thr} → 判重复" if J >= thr else f"J < {thr} → 复核后放弃（候选 ≠ 重复）"
        print(f"    (d{i}, d{j})：band {bands} 全等，签名相等 {eq}/{k}，精确 Jaccard {J:.3f}，{verdict}")
    print("  真实 Jaccard ≥ 0.5 却没进候选的对：")
    for (i, j), J in sorted(exact.items()):
        if J >= 0.5 and (i, j) not in cand:
            eq = int((sigs[i] == sigs[j]).sum())
            diff = [band for band, (ki, kj) in enumerate(zip(band_keys(sigs[i], b, r), band_keys(sigs[j], b, r))) if ki != kj]
            print(f"    (d{i}, d{j})：Jaccard {J:.3f}，签名相等 {eq}/{k}，但 3 个 band {diff} 各自至少有一列不等——没有一个 band 完整命中，"
                  f"理论上这对成候选的概率只有 1 − (1 − {J:.3f}²)³ = {p_candidate(J, b, r):.3f}")
    print("  漏掉不是 bug：候选阶段按概率命中，S 曲线在阈值附近本来就是一半一半；精确复核只能救候选里的假阳，救不了没进候选的对")
    print()


# ---------------- 5b. 理论公式 vs 实际 hash 族：候选概率量出来 ----------------
def exp_family():
    print("=== 5b. 候选概率：独立随机排列的理论值 vs 2000 次重跑实测 vs 实际 hash 族（k = 6, b = 3, r = 2）===")
    k, b, r, reps = 6, 3, 2, 2000
    sets = [word_shingles(d) for d in TRACE_DOCS]
    universe = {s: i for i, s in enumerate(sorted(set().union(*sets)))}
    pairs = [(0, 1), (1, 2), (0, 2), (3, 4)]
    rng = np.random.default_rng(0)
    hit_perm = {p: 0 for p in pairs}
    eq_perm = {p: 0 for p in pairs}
    hit_hash = {p: 0 for p in pairs}
    eq_hash = {p: 0 for p in pairs}
    hit_bad = {p: 0 for p in pairs}
    for t in range(reps):
        perms = np.array([rng.permutation(len(universe)) for _ in range(k)])
        sigs = np.array([perm_signature(S, perms, universe) for S in sets])
        cand = lsh_candidates(sigs, b, r)
        mh = MinHash(k, seed=10_000 + t)
        sigs_h = np.array([mh.signature(S) for S in sets])
        cand_h = lsh_candidates(sigs_h, b, r)
        bad = MinHash(k, seed=10_000 + t)
        bad.a[:] = 1                                                       # 只平移不乘：k 个 hash 高度相关
        sigs_b = np.array([bad.signature(S) for S in sets])
        cand_b = lsh_candidates(sigs_b, b, r)
        for p in pairs:
            hit_perm[p] += p in cand
            hit_hash[p] += p in cand_h
            hit_bad[p] += p in cand_b
            eq_perm[p] += int((sigs[p[0]] == sigs[p[1]]).sum())
            eq_hash[p] += int((sigs_h[p[0]] == sigs_h[p[1]]).sum())
    print(f"  {'对':<9}{'Jaccard':>8}{'签名相等率:排列':>14}{'仿射hash':>9}{'候选概率:理论':>13}{'排列实测':>9}{'仿射hash':>9}{'只平移的hash':>11}")
    for p in pairs:
        J = jaccard(sets[p[0]], sets[p[1]])
        print(f"  (d{p[0]}, d{p[1]})  {J:8.3f}{eq_perm[p] / (reps * k):14.3f}{eq_hash[p] / (reps * k):9.3f}"
              f"{p_candidate(J, b, r):13.3f}{hit_perm[p] / reps:9.3f}{hit_hash[p] / reps:9.3f}{hit_bad[p] / reps:11.3f}")
    # 图：把 Jaccard 从 0.2 扫到 0.95（两个 60 元集合、控制交集大小），三种签名的候选概率 vs 理论 S 曲线
    js = np.arange(0.2, 0.96, 0.05)
    rows: dict[str, list[float]] = {"perm": [], "hash": [], "bad": []}
    reps_fig = 300
    for J in js:
        shared = round(2 * 60 * J / (1 + J))
        A = {f"s{i}" for i in range(60)}
        B = {f"s{i}" for i in range(60 - shared, 120 - shared)}
        uni = {s: i for i, s in enumerate(sorted(A | B))}
        hits = {"perm": 0, "hash": 0, "bad": 0}
        for t in range(reps_fig):
            perms = np.array([rng.permutation(len(uni)) for _ in range(k)])
            sg = np.array([perm_signature(S, perms, uni) for S in (A, B)])
            hits["perm"] += (0, 1) in lsh_candidates(sg, b, r)
            mh = MinHash(k, seed=50_000 + t)
            sg = np.array([mh.signature(S) for S in (A, B)])
            hits["hash"] += (0, 1) in lsh_candidates(sg, b, r)
            mh.a[:] = 1
            sg = np.array([mh.signature(S) for S in (A, B)])
            hits["bad"] += (0, 1) in lsh_candidates(sg, b, r)
        for key, row in rows.items():
            row.append(hits[key] / reps_fig)
    ss = np.linspace(0.1, 1, 200)
    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    ax.plot(ss, p_candidate(ss, b, r), color=C["gray"], label="理论 1 − (1 − s²)³（独立随机排列）")
    ax.plot(js, rows["perm"], "o", ms=4, color=C["blue"], label="独立随机排列实测（每点 300 次）")
    ax.plot(js, rows["hash"], "s", ms=4, mfc="none", color=C["green"], label="仿射 hash (a·x + b) mod p 实测")
    ax.plot(js, rows["bad"], "^", ms=4, color=C["red"], label="只平移 (x + b) mod p：k 个 hash 高度相关")
    ax.plot(ss, ss, ":", color=C["red"], lw=0.8, label="P = s（k 列完全相关时的极限）")
    ax.set_xlabel("两个集合的真实 Jaccard")
    ax.set_ylabel("成为候选的概率")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    save(fig, "09-lsh-theory-vs-hash-family")
    print("  独立随机排列下，签名相等率 = Jaccard、候选概率 = 1 − (1 − J^r)^b 都是精确的期望；(a·x + b) mod p 这种仿射 hash 只是近似，"
          "这里与理论差在千分位；把 a 固定为 1 让 k 个 hash 高度相关，候选概率就塌回接近 J 本身——公式的前提是各列独立，hash 族不满足就不能照搬")
    print()


# ---------------- 5c. 近重复不传递：保留策略 ----------------
def union_find_groups(n, pairs):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return sorted(groups.values())


def greedy_keep(order, sets, thr):
    """按给定顺序扫一遍：与已保留的代表 Jaccard ≥ thr 就丢，否则自己成为代表。"""
    kept: list[int] = []
    for i in order:
        if all(jaccard(sets[i], sets[j]) < thr for j in kept):
            kept.append(i)
    return sorted(kept)


def exp_retain():
    print("=== 5c. 近重复不传递：d0 ≈ d1、d1 ≈ d2、d0 ≉ d2，保留谁 ===")
    thr = 0.7
    sets = [word_shingles(d) for d in TRACE_DOCS]
    exact = exact_pairs(sets)
    dup = sorted(p for p, J in exact.items() if J >= thr)
    print(f"  阈值 {thr} 下判为重复的对：{dup}；(d0, d2) 的 Jaccard {exact[(0, 2)]:.3f} 不够——重复关系不传递")
    groups = union_find_groups(len(sets), dup)
    print(f"  策略 A 连通分量（并查集）：分量 {groups}，每个分量留编号最小的 → 保留 {[g[0] for g in groups]}，d2 被删，虽然它和代表 d0 的 Jaccard 只有 {exact[(0, 2)]:.3f}")
    for order in ([0, 1, 2, 3, 4, 5], [1, 0, 2, 3, 4, 5], [2, 1, 0, 3, 4, 5]):
        kept = greedy_keep(order, sets, thr)
        print(f"  策略 B 贪心按顺序 {order}：保留 {kept}（只与已保留的代表比，删除的都与某个代表 J ≥ {thr}）")
    print("  连通分量删得多、保留集互不相似但可能删掉与代表并不相似的文本；贪心保留的每一条都离代表够远但结果依赖扫描顺序。"
          "两种都对，选哪种取决于你更怕漏删（训练集重复）还是误删（丢内容）——先定策略再谈去重率")
    # 放大：500 段随机文本 + 100 条「改写链」A → B → C（每步改 3 词），看两种策略删多少
    r = np.random.default_rng(5)
    vocab = [f"w{i}" for i in range(500)]
    docs = [" ".join(r.choice(vocab, 40)) for _ in range(500)]
    for _ in range(100):
        prev = docs[int(r.integers(0, 500))]
        for _step in range(2):
            w = prev.split()
            for j in r.choice(len(w), 3, replace=False):
                w[j] = str(r.choice(vocab))
            prev = " ".join(w)
            docs.append(prev)
    sets = [shingles(d, 5) for d in docs]
    exact = exact_pairs(sets, 0.3)
    dup = sorted(p for p, J in exact.items() if J >= thr)
    groups = union_find_groups(len(docs), dup)
    big = [g for g in groups if len(g) > 1]
    nontrans = 0
    for g in big:
        for x in range(len(g)):
            for y in range(x + 1, len(g)):
                if exact.get((g[x], g[y]), 0.0) < thr:
                    nontrans += 1
    kept_cc = sum(1 for _ in groups)
    kept_greedy = len(greedy_keep(list(range(len(docs))), sets, thr))
    kept_greedy_rev = len(greedy_keep(list(range(len(docs)))[::-1], sets, thr))
    print(f"  700 段（500 随机 + 100 条两步改写链）：J ≥ {thr} 的对 {len(dup)}，连通分量里 J < {thr} 的同组对 {nontrans} 个（不传递的证据）")
    print(f"  连通分量保留 {kept_cc} 段；贪心正序保留 {kept_greedy} 段、倒序保留 {kept_greedy_rev} 段——同一批重复对，三种答案")
    print()


EXPS = {"tiny": exp_tiny, "estimate": exp_estimate, "error": exp_error, "scurve": exp_scurve, "dedup": exp_dedup, "semantic": exp_semantic,
        "trace": exp_trace, "family": exp_family, "retain": exp_retain}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    for n in names:
        EXPS[n]()
