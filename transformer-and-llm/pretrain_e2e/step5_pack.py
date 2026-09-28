"""第 5 步：tokenize + 打包——把每篇文档变成 token 序列，末尾加 <|endoftext|>，首尾相接成一条长流，
按文档切 train / val（98 / 2），写成 uint16 的 train.bin / val.bin（nanoGPT 的格式）。
输出 out/e2e-5-pack.svg（文档 token 长度分布与一条 256 长的训练序列里有几篇文档）。

    python step5_pack.py
"""
import numpy as np
from tokenizers import Tokenizer

from _plot import C, plt, save
from common import DATA, Timer, fmt, read_jsonl

BLOCK = 256

if __name__ == "__main__":
    docs = read_jsonl(f"{DATA}/clean.jsonl")
    tok = Tokenizer.from_file(f"{DATA}/tokenizer.json")
    eos = tok.token_to_id("<|endoftext|>")
    with Timer("tokenize"):
        encs = tok.encode_batch([d["text"] for d in docs])
    lens = np.array([len(e.ids) for e in encs])
    chars = np.array([len(d["text"]) for d in docs])
    print(f"{fmt(len(docs))} 篇 → {fmt(int(lens.sum()))} 个 token（不含 eos），{chars.sum()/lens.sum():.2f} 字符/token")
    print(f"每篇 token 数：中位数 {int(np.median(lens))}，均值 {lens.mean():.0f}，最短 {lens.min()}，最长 {fmt(int(lens.max()))}；{(lens < BLOCK).mean():.0%} 的文档不到 {BLOCK} 个 token")

    rng = np.random.default_rng(0)
    order = rng.permutation(len(docs))
    n_val = max(1, int(len(docs) * 0.02))
    val_ids, train_ids = order[:n_val], order[n_val:]
    def stream(ids):
        out = []
        for i in ids:
            out.extend(encs[i].ids); out.append(eos)
        return np.array(out, dtype=np.uint16)
    with Timer("打包"):
        train = stream(train_ids); val = stream(val_ids)
    train.tofile(f"{DATA}/train.bin"); val.tofile(f"{DATA}/val.bin")
    print(f"train.bin：{fmt(len(train))} 个 token（{len(train_ids)} 篇，{train.nbytes/1e6:.1f} MB）；val.bin：{fmt(len(val))} 个 token（{n_val} 篇）")
    print(f"按 block_size = {BLOCK} 切：train 有 {fmt(len(train)//BLOCK)} 条不重叠的序列；一条序列平均含 {BLOCK / (lens.mean() + 1):.2f} 个文档边界（跨过边界时 attention 会看到上一篇无关的内容——文中「文档掩码」的取舍）")
    # 一条序列里有几个 eos
    n_eos = [(train[i:i + BLOCK] == eos).sum() for i in range(0, min(len(train), 200_000) - BLOCK, BLOCK)]
    print(f"前 {len(n_eos)} 条训练序列里 eos 的个数：0 个的占 {np.mean(np.array(n_eos)==0):.0%}，1 个 {np.mean(np.array(n_eos)==1):.0%}，≥ 2 个 {np.mean(np.array(n_eos)>=2):.0%}")
    print("打包示例（第 3 条序列，解码回文本，⏎ 是换行，<|endoftext|> 是文档边界）：")
    print("   " + tok.decode(train[2 * BLOCK:3 * BLOCK].tolist(), skip_special_tokens=False).replace("\n", " ⏎ ")[:500])

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.7))
    axes[0].hist(np.log10(lens), bins=40, color=C["blue"])
    axes[0].axvline(np.log10(BLOCK), color=C["red"], ls="--", lw=1); axes[0].text(np.log10(BLOCK) + 0.05, axes[0].get_ylim()[1] * 0.9, f"block_size {BLOCK}", color=C["red"], fontsize=8)
    axes[0].set_xticks([1.5, 2, 2.5, 3, 3.5, 4], ["30", "100", "300", "1K", "3K", "10K"])
    axes[0].set(xlabel="每篇文档的 token 数（对数刻度）", ylabel="文档数", title="文档长短不一，所以要打包")
    vals, cnts = np.unique(np.minimum(n_eos, 5), return_counts=True)
    axes[1].bar([str(v) if v < 5 else "≥5" for v in vals], cnts / cnts.sum(), color=C["green"])
    axes[1].set(xlabel=f"一条 {BLOCK} token 的训练序列里的文档边界数", ylabel="比例", title="约四分之一的序列跨了文档边界")
    save(fig, "e2e-5-pack")
