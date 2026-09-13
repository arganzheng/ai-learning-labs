"""BPE 从零实现（Transformer 与 LLM 09）：byte-level BPE 的训练与编码，词表大小 → 压缩率曲线。
https://arganzheng.life/tokenizer-vocabulary-and-token-efficiency.html

纯 Python 标准库。语料用 Python 自带的标准库源码（不需要下载），
可选 --corpus 指向任意文本文件。

    python bpe_from_scratch.py            # 玩具例子 + 标准库语料上的词表大小扫描
    python bpe_from_scratch.py --quick    # 只跑玩具例子与 2 个词表大小
"""
import argparse
import re
import sys
import sysconfig
from collections import Counter
from pathlib import Path

# GPT-2 风格的预分词：把文本先切成"词"（带前导空格）、数字串、标点，BPE 只在词内合并。
# 这一步决定了 token 永远不会跨越空格和标点（"of the" 不会变成一个 token）。
PRETOKEN_RE = re.compile(r" ?[A-Za-z_]+| ?\d+| ?[^\sA-Za-z_\d]+|\s+")


def pretokenize(text):
    return PRETOKEN_RE.findall(text)


def train_bpe(text, vocab_size, verbose_steps=0):
    """返回 merges（按学习顺序的 (a, b) 列表）。基础词表是 256 个字节。"""
    words = Counter(tuple(w.encode("utf-8")) for w in pretokenize(text))
    merges = []
    n_merges = vocab_size - 256
    for step in range(n_merges):
        pairs = Counter()
        for word, freq in words.items():
            for a, b in zip(word, word[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        (a, b), freq = pairs.most_common(1)[0]
        new_id = 256 + step
        merges.append((a, b))
        if step < verbose_steps:
            print(f"  merge {step + 1:>3}: {show(a, merges)!r:>10} + {show(b, merges)!r:<10} "
                  f"-> id {new_id}  (出现 {freq} 次)")
        merged = {}
        for word, freq in words.items():
            out, i = [], 0
            while i < len(word):
                if i + 1 < len(word) and word[i] == a and word[i + 1] == b:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            merged[tuple(out)] = merged.get(tuple(out), 0) + freq
        words = merged
    return merges


def show(tok_id, merges):
    """把一个 token id 还原成字节串（供打印；不可解码的字节用 \\x 转义）。"""
    return bytes(expand(tok_id, merges)).decode("utf-8", errors="backslashreplace")


def expand(tok_id, merges):
    if tok_id < 256:
        return [tok_id]
    a, b = merges[tok_id - 256]
    return expand(a, merges) + expand(b, merges)


def encode(text, merges):
    """编码：对每个预分词后的词，按 merges 的学习顺序反复合并。"""
    rank = {pair: i for i, pair in enumerate(merges)}
    ids = []
    for w in pretokenize(text):
        word = list(w.encode("utf-8"))
        while len(word) > 1:
            best = min(zip(word, word[1:]), key=lambda p: rank.get(p, 1 << 30))
            if best not in rank:
                break
            new_id, out, i = 256 + rank[best], [], 0
            while i < len(word):
                if i + 1 < len(word) and (word[i], word[i + 1]) == best:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            word = out
        ids.extend(word)
    return ids


def stdlib_corpus(max_bytes):
    root = Path(sysconfig.get_paths()["stdlib"])
    chunks, size = [], 0
    for p in sorted(root.glob("*.py")):
        s = p.read_text(errors="ignore")
        chunks.append(s)
        size += len(s.encode())
        if size >= max_bytes:
            break
    return "".join(chunks)


def toy_example():
    print("=== 玩具例子：经典的 low / lower / newest / widest ===")
    text = " ".join(["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3)
    merges = train_bpe(text, 256 + 8, verbose_steps=8)
    for w in ["lowest", "newer", "wide"]:
        ids = encode(w, merges)
        print(f"  encode({w!r}) -> {[show(i, merges) for i in ids]}")
    print()


def vocab_sweep(text, sizes, samples):
    print(f"=== 词表大小扫描（语料 {len(text.encode()) / 1e6:.1f} MB 标准库源码）===")
    print(f"{'vocab':>7} {'训练集 bytes/token':>16} " + " ".join(f"{n:>14}" for n in samples))
    text_bytes = len(text.encode())
    largest = train_bpe(text, max(sizes))
    for v in sizes:
        merges = largest[: v - 256]
        n = len(encode(text, merges))
        cells = []
        for _, s in samples.items():
            ids = encode(s, merges)
            cells.append(f"{len(s.encode()) / len(ids):>8.2f} B/tok")
        print(f"{v:>7} {text_bytes / n:>16.2f} " + " ".join(f"{c:>14}" for c in cells))
    return largest


SAMPLES = {
    "英文": "The quick brown fox jumps over the lazy dog because it was late for dinner.",
    "Python": "def forward(self, x):\n        return self.norm(x + self.attn(x))\n",
    "中文": "机器学习是人工智能的一个分支，研究计算机怎样从数据中学习规律。",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--corpus", help="用这个文本文件代替标准库源码做训练语料")
    ap.add_argument("--corpus-mb", type=float, default=1.0)
    args = ap.parse_args()

    toy_example()
    text = Path(args.corpus).read_text(errors="ignore") if args.corpus else stdlib_corpus(int(args.corpus_mb * 1e6))
    if args.quick:
        text, sizes = text[: 300_000], [512, 2048]
    else:
        sizes = [256, 512, 1024, 2048, 4096, 8192, 16384]
    merges = vocab_sweep(text, sizes, SAMPLES)
    print()
    print("=== 中文在英文/代码语料上训出的词表里：几乎回到字节级 ===")
    ids = encode(SAMPLES["中文"], merges)
    print(f"  {len(SAMPLES['中文'])} 个字符 -> {len(ids)} 个 token（每字符 {len(ids) / len(SAMPLES["中文"]):.1f} token；UTF-8 每个汉字 3 字节）")
    print("  前 12 个 token:", [show(i, merges) for i in ids[:12]])


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
