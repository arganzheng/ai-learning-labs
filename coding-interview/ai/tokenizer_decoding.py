"""面试手撕代码（16）：手撕 tokenizer 与解码 —— BPE、采样、beam search、投机解码。

https://arganzheng.life/coding-interview-tokenizer-and-decoding.html

    python tokenizer_decoding.py            # 示例输出
    python tokenizer_decoding.py --check    # 与 torch / 数学性质对拍
"""
from __future__ import annotations

import argparse
import math
import random
from collections import Counter

import numpy as np

from attention import softmax, log_softmax


# ---------- 1. BPE ----------

def bpe_train(corpus: list[str], num_merges: int) -> list[tuple[str, str]]:
    """字节级简化版：每个词拆成字符 + 词尾 '</w>'，反复合并出现次数最多的相邻对。返回 merge 表（有序）。"""
    words = Counter(tuple(w) + ("</w>",) for w in corpus)        # 词 -> 频次，词表示为符号元组
    merges: list[tuple[str, str]] = []
    for _ in range(num_merges):
        pairs: Counter = Counter()
        for sym, freq in words.items():
            for a, b in zip(sym, sym[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))           # 频次相同按字典序，保证确定性
        merges.append(best)
        words = Counter({_merge(sym, best): f for sym, f in words.items()})
    return merges


def _merge(sym: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    out, i = [], 0
    while i < len(sym):
        if i + 1 < len(sym) and (sym[i], sym[i + 1]) == pair:
            out.append(sym[i] + sym[i + 1])
            i += 2
        else:
            out.append(sym[i])
            i += 1
    return tuple(out)


def bpe_encode(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """按 merge 表的顺序（= 训练时的优先级）依次应用。"""
    sym = tuple(word) + ("</w>",)
    rank = {p: i for i, p in enumerate(merges)}
    while len(sym) > 1:
        cands = [(rank[(a, b)], (a, b)) for a, b in zip(sym, sym[1:]) if (a, b) in rank]
        if not cands:
            break
        _, pair = min(cands)                                     # 优先级最高（rank 最小）的对
        sym = _merge(sym, pair)
    return list(sym)


# ---------- 2. 采样 ----------

def sample_logits(logits: np.ndarray, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0,
                  min_p: float = 0.0, rng: np.random.Generator | None = None) -> int:
    """从一行 logits 采一个 token。顺序：temperature → top-k → top-p → min-p → 归一化 → 采样。"""
    rng = rng or np.random.default_rng()
    if temperature == 0:
        return int(logits.argmax())                              # 贪心
    logits = logits / temperature
    probs = softmax(logits)
    if top_k > 0 and top_k < len(probs):
        kth = np.sort(probs)[-top_k]
        probs = np.where(probs >= kth, probs, 0.0)
    if top_p < 1.0:
        order = np.argsort(-probs)
        cum = np.cumsum(probs[order])
        cutoff = order[cum - probs[order] >= top_p]              # 累计到达 top_p 之后的全部丢掉（保留首个越界项）
        probs[cutoff] = 0.0
    if min_p > 0:
        probs = np.where(probs >= min_p * probs.max(), probs, 0.0)
    probs = probs / probs.sum()
    return int(rng.choice(len(probs), p=probs))


def apply_repetition_penalty(logits: np.ndarray, generated: list[int], penalty: float) -> np.ndarray:
    """HF 的写法：已生成 token 的 logit，正的除以 penalty、负的乘以 penalty（都朝更不可能推）。"""
    out = logits.copy()
    for t in set(generated):
        out[t] = out[t] / penalty if out[t] > 0 else out[t] * penalty
    return out


# ---------- 3. beam search ----------

def beam_search(step_fn, bos: int, eos: int, beam: int, max_len: int, length_alpha: float = 0.0):
    """step_fn(prefix: list[int]) -> log_probs (V,)。返回 (最佳序列, 得分)。
    length_alpha > 0 时用长度归一化 score / len^alpha，抑制"越短越好"。"""
    beams = [([bos], 0.0)]
    finished = []
    for _ in range(max_len):
        cands = []
        for seq, score in beams:
            lp = step_fn(seq)
            for t in np.argpartition(-lp, beam)[:beam]:          # 每个 beam 只取前 beam 个，够用
                cands.append((seq + [int(t)], score + float(lp[t])))
        cands.sort(key=lambda c: -c[1])
        beams = []
        for seq, score in cands:
            if seq[-1] == eos:
                finished.append((seq, score))
            else:
                beams.append((seq, score))
            if len(beams) == beam:
                break
        if not beams:
            break
    finished.extend(beams)
    norm = lambda s: s[1] / (len(s[0]) ** length_alpha) if length_alpha else s[1]
    return max(finished, key=norm)


# ---------- 4. 蓄水池抽样 ----------

def reservoir_sample(stream, k: int, rng: random.Random):
    """流里每个元素等概率进入大小 k 的样本：第 i 个元素（1-based）以 k/i 的概率替换随机一个。"""
    res = []
    for i, x in enumerate(stream, 1):
        if i <= k:
            res.append(x)
        else:
            j = rng.randint(1, i)
            if j <= k:
                res[j - 1] = x
    return res


# ---------- 5. 投机解码的接受规则 ----------

def speculative_accept(p_draft: np.ndarray, p_target: np.ndarray, draft_token: int, rng: np.random.Generator) -> tuple[bool, int]:
    """一步：草稿模型采出 draft_token（分布 q），目标模型分布 p。
    以 min(1, p[x]/q[x]) 接受；拒绝则从 norm(max(0, p - q)) 重采一个。返回 (是否接受, 最终 token)。
    这样得到的 token 分布严格等于 p。"""
    q, p = p_draft, p_target
    if rng.random() < min(1.0, p[draft_token] / q[draft_token]):
        return True, draft_token
    resid = np.maximum(p - q, 0)
    resid /= resid.sum()
    return False, int(rng.choice(len(p), p=resid))


# ---------- demo & check ----------

def demo():
    print("== BPE：在小语料上训练 10 次合并")
    corpus = ["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3
    merges = bpe_train(corpus, 10)
    for i, m in enumerate(merges):
        print(f"  merge {i}: {m[0]!r} + {m[1]!r} -> {m[0] + m[1]!r}")
    for w in ("lowest", "newer", "wide"):
        print(f"  encode {w!r:10s} -> {bpe_encode(w, merges)}")

    print("\n== 采样：同一行 logits 在不同策略下的分布（10000 次统计）")
    rng = np.random.default_rng(0)
    logits = np.array([2.0, 1.5, 1.0, 0.0, -1.0])
    for name, kw in [("T=1", {}), ("T=0.5", {"temperature": 0.5}), ("top_k=2", {"top_k": 2}), ("top_p=0.8", {"top_p": 0.8}), ("min_p=0.3", {"min_p": 0.3})]:
        cnt = Counter(sample_logits(logits, rng=rng, **kw) for _ in range(10000))
        print(f"  {name:10s}", [f"{cnt[i] / 10000:.3f}" for i in range(5)])
    print(f"  softmax(logits) =", np.round(softmax(logits), 3).tolist())

    print("\n== beam search 在一个玩具 LM 上（V=4，eos=3）")
    table = {(): [0.1, 0.5, 0.4, 0.0], (1,): [0.3, 0.3, 0.1, 0.3], (2,): [0.02, 0.02, 0.01, 0.95], (1, 0): [0.0, 0.0, 0.0, 1.0]}
    def step_fn(prefix):
        key = tuple(prefix[1:])
        probs = np.array(table.get(key, [0.0, 0.0, 0.0, 1.0]))
        return np.log(probs + 1e-12)
    for b in (1, 2):
        seq, score = beam_search(step_fn, bos=-1, eos=3, beam=b, max_len=4)
        print(f"  beam={b}: seq={seq[1:]}, log p={score:.3f}, p={math.exp(score):.3f}")

    print("\n== 蓄水池抽样：k=3，流长 10，每个元素被选中的频率（20000 次）")
    r = random.Random(0)
    cnt = Counter()
    for _ in range(20000):
        for x in reservoir_sample(range(10), 3, r):
            cnt[x] += 1
    print("  ", [f"{cnt[i] / 20000:.3f}" for i in range(10)], "（理论 0.3）")

    print("\n== 投机解码：接受-拒绝后 token 分布应等于目标分布 p")
    q = np.array([0.5, 0.3, 0.2]); p = np.array([0.2, 0.3, 0.5])
    cnt = Counter(); acc = 0
    for _ in range(30000):
        x = int(rng.choice(3, p=q))
        ok, t = speculative_accept(q, p, x, rng)
        acc += ok; cnt[t] += 1
    print(f"   最终分布 {[f'{cnt[i] / 30000:.3f}' for i in range(3)]}（目标 p = {p.tolist()}）；接受率 {acc / 30000:.3f}（理论 1 - TV(p,q) = {1 - 0.5 * np.abs(p - q).sum():.3f}）")


def check():
    import torch
    rng = np.random.default_rng(3)

    # BPE：encode 后 join 还原原词
    corpus = ["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3
    merges = bpe_train(corpus, 15)
    for w in ("lowest", "newer", "wide", "x"):
        assert "".join(bpe_encode(w, merges)) == w + "</w>"

    # 采样各策略的支撑集
    logits = rng.standard_normal(20)
    probs = softmax(logits)
    counts = Counter(sample_logits(logits, top_k=3, rng=rng) for _ in range(3000))
    assert set(counts) <= set(np.argsort(-probs)[:3].tolist())
    counts = Counter(sample_logits(logits, top_p=0.5, rng=rng) for _ in range(3000))
    order = np.argsort(-probs); cum = np.cumsum(probs[order])
    allowed = set(order[:int(np.searchsorted(cum, 0.5)) + 1].tolist())
    assert set(counts) <= allowed, (set(counts), allowed)
    # temperature：与 torch 的 softmax(logits / T) 一致
    for T in (0.5, 2.0):
        assert np.allclose(softmax(logits / T), torch.softmax(torch.tensor(logits) / T, -1).numpy())

    # repetition penalty：已生成 token 的概率必然下降
    pen = apply_repetition_penalty(logits, [0, 1], 1.5)
    assert softmax(pen)[0] < probs[0] and softmax(pen)[1] < probs[1]

    # beam=1 等于贪心
    table = {(): [0.1, 0.6, 0.3, 0.0], (1,): [0.5, 0.1, 0.1, 0.3], (1, 0): [0.0, 0.0, 0.0, 1.0]}
    step = lambda pre: np.log(np.array(table.get(tuple(pre[1:]), [0.0, 0.0, 0.0, 1.0])) + 1e-12)
    greedy = [-1]
    while greedy[-1] != 3 and len(greedy) < 5:
        greedy.append(int(step(greedy).argmax()))
    assert beam_search(step, -1, 3, 1, 4)[0] == greedy

    # 蓄水池：均匀性（卡方粗检）
    r = random.Random(1)
    cnt = Counter()
    N = 20000
    for _ in range(N):
        for x in reservoir_sample(range(10), 3, r):
            cnt[x] += 1
    assert all(abs(cnt[i] / N - 0.3) < 0.02 for i in range(10))

    # 投机解码：最终分布 == p
    q = np.array([0.5, 0.3, 0.2]); p = np.array([0.2, 0.3, 0.5])
    cnt = Counter()
    N = 60000
    for _ in range(N):
        _, t = speculative_accept(q, p, int(rng.choice(3, p=q)), rng)
        cnt[t] += 1
    assert all(abs(cnt[i] / N - p[i]) < 0.01 for i in range(3))
    print("tokenizer_decoding.py: all checks passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    check() if args.check else demo()
