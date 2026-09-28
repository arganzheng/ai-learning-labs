"""第 4 步：训 tokenizer——在清洗后的语料上训一个 byte-level BPE（词表 4096），看它学到的合并、
词表大小对压缩率（字符/token）的影响，以及与 GPT-2 tokenizer 的对比。
输出 data/tokenizer.json 与 out/e2e-4-tokenizer.svg。

    python step4_tokenizer.py
"""
import os
import re
import sys

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from _plot import C, plt, save
from common import DATA, Timer, fmt, read_jsonl

VOCAB = 4096
SAMPLE = ("The board requires you to be registered and logged in to view profiles. "
          "Residential and commercial roofing contractors in Pennsylvania offer free estimates.\n"
          "def tokenize(text):\n    return text.split()\n"
          "北京是中华人民共和国的首都。")


def train(texts, vocab_size):
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)   # GPT-2 同款：先按空格 / 标点粗切，再在字节上合并
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts, trainer)
    return tok


if __name__ == "__main__":
    docs = read_jsonl(f"{DATA}/clean.jsonl")
    texts = [d["text"] for d in docs]
    total_chars = sum(len(t) for t in texts)
    print(f"语料：{fmt(len(texts))} 篇，{fmt(total_chars)} 字符")

    with Timer(f"训 BPE 词表 {VOCAB}"):
        tok = train(texts, VOCAB)
    tok.save(f"{DATA}/tokenizer.json")
    vocab = tok.get_vocab()
    inv = {v: k for k, v in vocab.items()}
    merges_shown = [inv[i] for i in range(257, 257 + 30)]
    print("最早学到的 30 个合并（256 个字节之后的词表项，Ġ 表示前面有空格）：", " ".join(merges_shown))
    longest = sorted(vocab, key=len, reverse=True)[:12]
    print("最长的词表项：", " ".join(longest))

    enc = tok.encode(SAMPLE)
    print(f"\n一段样本（{len(SAMPLE)} 字符）→ {len(enc.ids)} 个 token，{len(SAMPLE)/len(enc.ids):.2f} 字符/token：")
    print("   " + " | ".join(t.replace("Ġ", "␣").replace("Ċ", "⏎") for t in enc.tokens))
    print("   中文部分：每个汉字是 3 个字节，语料里几乎没有中文、词表没学过它们的合并 → 退回字节级，一个字要 3 个 token")

    # 词表大小扫描：压缩率
    sub = texts[: max(200, len(texts) // 5)]
    sub_chars = sum(len(t) for t in sub)
    sizes = [256 + 8, 512, 1024, 2048, 4096, 8192, 16384]
    ratio = []
    print("\n词表大小 → 字符/token（在语料的一个子集上）：")
    for v in sizes:
        t = train(texts, v) if v != VOCAB else tok
        n_tok = sum(len(e.ids) for e in t.encode_batch(sub))
        ratio.append(sub_chars / n_tok)
        print(f"   V = {v:>6}: {ratio[-1]:.2f} 字符/token，{fmt(n_tok)} 个 token")
    # GPT-2 对照
    try:
        import tiktoken
        g = tiktoken.get_encoding("gpt2")
        n_gpt2 = sum(len(g.encode(t)) for t in sub)
        print(f"   GPT-2 tokenizer（V = 50257，在 WebText 上训的）：{sub_chars/n_gpt2:.2f} 字符/token")
    except Exception as e:  # noqa: BLE001
        n_gpt2 = None
        print(f"   （GPT-2 tokenizer 不可用：{e}）")

    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    ax.plot(sizes, ratio, "o-", color=C["blue"], label="在本语料上训的 BPE")
    if n_gpt2:
        ax.axhline(sub_chars / n_gpt2, color=C["orange"], ls="--", lw=1, label="GPT-2 tokenizer（V = 50257）")
    ax.axvline(VOCAB, color=C["gray"], ls=":", lw=1); ax.text(VOCAB * 1.1, ratio[0] + 0.1, f"本文用 V = {VOCAB}", fontsize=8, color=C["gray"])
    ax.set(xscale="log", xlabel="词表大小 V（对数）", ylabel="字符 / token（越高越省 token）", title="词表越大压缩率越高，但边际收益递减")
    ax.set_xticks(sizes, [str(s) for s in sizes], fontsize=7)
    ax.legend(fontsize=7)
    save(fig, "e2e-4-tokenizer")
