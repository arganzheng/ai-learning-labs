"""MinHash + LSH 近重复去重（Transformer 与 LLM 11）：从零实现 FineWeb 用的配置（5-gram、112 个哈希、14 桶 × 8 行），
验证"两个文档进同一个桶的概率"随 Jaccard 相似度的 S 曲线，并在一组人造近重复文档上跑一遍。
https://arganzheng.life/pretraining-data-pipeline-dedup-filtering-and-mixture.html

纯 Python 标准库。  python minhash_lsh.py [--quick]
"""
import argparse
import hashlib
import random
import re
from collections import defaultdict

P = (1 << 61) - 1          # 梅森素数，做 (a·x + b) mod P 的通用哈希
N_HASH, BANDS = 112, 14     # FineWeb：112 个哈希函数分成 14 个桶，每桶 8 行
ROWS = N_HASH // BANDS
NGRAM = 5


def shingles(text, n=NGRAM):
    """按词切 n-gram，转成 64 位整数集合。"""
    words = re.findall(r"\w+", text.lower())
    out = set()
    for i in range(max(1, len(words) - n + 1)):
        g = " ".join(words[i : i + n])
        out.add(int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big"))
    return out


class MinHasher:
    def __init__(self, n_hash=N_HASH, seed=0):
        rng = random.Random(seed)
        self.ab = [(rng.randrange(1, P), rng.randrange(0, P)) for _ in range(n_hash)]

    def signature(self, sh):
        """对每个哈希函数取集合里的最小值：P[min_h(A) == min_h(B)] = Jaccard(A, B)。"""
        return [min((a * x + b) % P for x in sh) for a, b in self.ab]


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def lsh_candidates(sigs, bands=BANDS, rows=ROWS):
    """把签名切成 bands 段，每段 rows 个值作 key；任一段完全相同的两篇文档成为候选对。"""
    buckets = defaultdict(list)
    for doc_id, sig in sigs.items():
        for b in range(bands):
            key = (b, tuple(sig[b * rows : (b + 1) * rows]))
            buckets[key].append(doc_id)
    pairs = set()
    for ids in buckets.values():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.add((min(ids[i], ids[j]), max(ids[i], ids[j])))
    return pairs


def p_candidate(j, bands=BANDS, rows=ROWS):
    """理论：P(至少一个桶全同) = 1 - (1 - J^rows)^bands。"""
    return 1 - (1 - j ** rows) ** bands


def make_pair_with_jaccard(target_j, size, rng):
    """构造两个整数集合，Jaccard 恰好接近 target_j。"""
    shared = int(size * 2 * target_j / (1 + target_j))
    base = set(rng.sample(range(1 << 40), 2 * size))
    common = set(list(base)[:shared])
    rest = list(base - common)
    a = common | set(rest[: size - shared])
    b = common | set(rest[size - shared : 2 * (size - shared)])
    return a, b


def s_curve_experiment(trials, rng, mh):
    print(f"=== S 曲线：{N_HASH} 个哈希，{BANDS} 桶 × {ROWS} 行；理论阈值 (1/b)^(1/r) = {(1 / BANDS) ** (1 / ROWS):.2f} ===")
    print(f"{'Jaccard':>8}{'理论 P(候选)':>13}{'实测':>8}{'签名估计的 J':>13}")
    for j in [0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        hits, est = 0, 0.0
        for _ in range(trials):
            a, b = make_pair_with_jaccard(j, 300, rng)
            sa, sb = mh.signature(a), mh.signature(b)
            est += sum(x == y for x, y in zip(sa, sb)) / N_HASH
            hits += any(sa[k * ROWS : (k + 1) * ROWS] == sb[k * ROWS : (k + 1) * ROWS] for k in range(BANDS))
        print(f"{j:>8.2f}{p_candidate(j):>13.3f}{hits / trials:>8.2f}{est / trials:>13.3f}")
    print("  阈值以下几乎不会成为候选，以上几乎必然：LSH 把 O(n²) 的两两比较变成一次分桶。\n")


DOCS = {
    "A 原文": "The transformer architecture relies on self attention to model long range dependencies in sequences. "
              "Each layer computes queries keys and values from the input and mixes information across positions. "
              "Residual connections and layer normalization keep training stable as depth increases.",
    "B 改了几个词": "The transformer architecture relies on self attention to capture long range dependencies in sequences. "
                 "Each layer computes queries keys and values from the input and mixes information across positions. "
                 "Residual connections and layer normalization keep optimization stable as depth increases.",
    "C 加了导航栏": "Home | Products | Blog | Contact us | Sign in. "
                 "The transformer architecture relies on self attention to model long range dependencies in sequences. "
                 "Each layer computes queries keys and values from the input and mixes information across positions. "
                 "Residual connections and layer normalization keep training stable as depth increases. "
                 "Copyright 2024 All rights reserved. Privacy policy. Terms of service.",
    "D 同主题另写": "Attention lets a network relate every token to every other token regardless of distance. "
                 "Stacking such layers with feed forward blocks yields the transformer, whose depth is made trainable "
                 "by residual paths and normalization.",
    "E 无关": "Sourdough starter needs regular feeding with flour and water; keep it at room temperature and discard half before each feed.",
}


def document_demo(mh):
    print("=== 五篇文档：真实 Jaccard、签名估计、是否成为 LSH 候选 ===")
    sh = {k: shingles(v) for k, v in DOCS.items()}
    sigs = {k: mh.signature(v) for k, v in sh.items()}
    cands = lsh_candidates(sigs)
    names = list(DOCS)
    print(f"{'对':<22}{'Jaccard(5-gram)':>16}{'签名估计':>10}{'LSH 候选':>9}")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            est = sum(x == y for x, y in zip(sigs[a], sigs[b])) / N_HASH
            flag = "是" if (min(a, b), max(a, b)) in cands else "否"
            print(f"{a[0] + '-' + b[0]:<22}{jaccard(sh[a], sh[b]):>16.2f}{est:>10.2f}{flag:>9}")
    print("  B 只改了 3 个词，5-gram 的 Jaccard 就掉到 0.57——每个改动打掉 5 个 n-gram，42 个词的短文档经不起几处改动；"
          "C 只是多了导航栏与版权行，Jaccard 0.68，也没过 0.72 的阈值；D 同主题但另写，Jaccard ≈ 0。\n")


def length_demo(mh):
    """同样的改动放到一篇长文档上：n-gram 被打掉的比例小得多，Jaccard 留在阈值之上。"""
    import re as _re
    long_text = _re.sub(r"\s+", " ", _re.__doc__)          # re 模块的文档字符串，约 1000 词
    words = long_text.split()
    edited = list(words)
    for k in (10, len(words) // 2, len(words) - 10):
        edited[k] = "CHANGED"
    with_chrome = "Home | Products | Blog | Contact us | Sign in. " + long_text + " Copyright 2024 All rights reserved. Privacy policy."
    docs = {"L 长文原文": long_text, "M 长文改 3 词": " ".join(edited), "N 长文加导航栏": with_chrome}
    sh = {k: shingles(v) for k, v in docs.items()}
    sigs = {k: mh.signature(v) for k, v in sh.items()}
    cands = lsh_candidates(sigs)
    print(f"=== 同样的改动放到 {len(words)} 个词的长文档上 ===")
    for a, b in [("L 长文原文", "M 长文改 3 词"), ("L 长文原文", "N 长文加导航栏")]:
        flag = "是" if (min(a, b), max(a, b)) in cands else "否"
        print(f"  {a} vs {b}: Jaccard {jaccard(sh[a], sh[b]):.2f}  LSH 候选 {flag}")
    print("  近重复检测的灵敏度依赖文档长度：网页正文通常几百到几千词，改几处、加个页眉页脚都还是近重复；短文档要用更低的阈值或更小的 n。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    rng = random.Random(0)
    mh = MinHasher()
    s_curve_experiment(40 if args.quick else 200, rng, mh)
    document_demo(mh)
    length_demo(mh)


if __name__ == "__main__":
    main()
