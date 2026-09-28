"""预训练（04）「先讲明白」：数据配比是取舍——两个域（第一篇的英文网页 + Python 标准库源码），代码占 0 / 25 / 50 / 75 / 100%
各训一个同样大小、同样步数的小模型，看两个域各自的 val loss 随配比怎么变。输出 out/e2e-mixture.svg。

    python domain_mixture.py            # 5 次训练，MPS 约 12 分钟
    python domain_mixture.py --quick    # 步数减到 1/4
    python domain_mixture.py --plot-only  # 只从 data/mixture.json 重画图
依赖：step4 / step5 已跑过（data/tokenizer.json、data/train.bin、data/val.bin）。
"""
import contextlib
import glob
import io
import json
import os
import sys
import sysconfig

import numpy as np
import torch
from tokenizers import Tokenizer

from _plot import C, plt, save
from common import DATA, fmt
from trainer import BLOCK, DEV, make_model, n_params, train

QUICK = "--quick" in sys.argv
STEPS = 200 if QUICK else 800
BS = 64
MIXES = [0.0, 0.25, 0.5, 0.75, 1.0]           # batch 里代码 token 的比例
CODE_CHARS = 30_000_000


def build_code_tokens(tok):
    """Python 标准库源码 → token 流（用第一篇在英文网页上训的 tokenizer；代码在它眼里很\"贵\"，文中会讨论）。"""
    cache = f"{DATA}/code_tokens.npy"
    if os.path.exists(cache):
        return np.load(cache)
    files = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "**", "*.py"), recursive=True))
    texts, n = [], 0
    for f in files:
        if "/test" in f or "site-packages" in f:
            continue
        try:
            s = open(f, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        texts.append(s); n += len(s)
        if n >= CODE_CHARS:
            break
    eos = tok.token_to_id("<|endoftext|>")
    ids = []
    for enc in tok.encode_batch(texts):
        ids.extend(enc.ids); ids.append(eos)
    arr = np.array(ids, dtype=np.uint16)
    np.save(cache, arr)
    print(f"代码语料：{len(texts)} 个 .py 文件，{fmt(n)} 字符 → {fmt(len(arr))} token（{n/len(arr):.2f} 字符/token；网页语料是 3.46——网页上训的词表切代码要多花一倍的 token）")
    return arr


class MixData:
    """两条 token 流按比例拼一个 batch；val 分别在两个域上算。"""
    def __init__(self, web, code, p_code, block=BLOCK):
        self.web, self.code, self.p, self.block = web, code, p_code, block
        self.pos = {"web": 0, "code": 0}
        self.g = torch.Generator().manual_seed(0)

    def _seq(self, arr, key, k):
        n_seq = (len(arr) - 1) // self.block
        if self.pos[key] + k > n_seq:
            self.pos[key] = 0
        starts = torch.arange(self.pos[key], self.pos[key] + k) * self.block
        self.pos[key] += k
        x = torch.stack([torch.from_numpy(arr[s:s + self.block].astype(np.int64)) for s in starts])
        y = torch.stack([torch.from_numpy(arr[s + 1:s + 1 + self.block].astype(np.int64)) for s in starts])
        return x, y

    def batch(self, split, bs):
        if split == "train":
            k_code = int(round(bs * self.p)); k_web = bs - k_code
            xs, ys = [], []
            if k_web:
                x, y = self._seq(self.web["train"], "web", k_web); xs.append(x); ys.append(y)
            if k_code:
                x, y = self._seq(self.code["train"], "code", k_code); xs.append(x); ys.append(y)
            return torch.cat(xs).to(DEV), torch.cat(ys).to(DEV)
        arr = self.web["val"] if split == "val" else self.code["val"]
        starts = torch.randint(len(arr) - self.block - 1, (bs,), generator=self.g)
        x = torch.stack([torch.from_numpy(arr[s:s + self.block].astype(np.int64)) for s in starts])
        y = torch.stack([torch.from_numpy(arr[s + 1:s + 1 + self.block].astype(np.int64)) for s in starts])
        return x.to(DEV), y.to(DEV)

    @property
    def n_train_tokens(self):
        return len(self.web["train"])


@torch.no_grad()
def domain_loss(model, data, split, batches=32, bs=32):
    model.eval()
    ls = [model(*data.batch(split, bs))[1].item() for _ in range(batches)]
    model.train()
    return float(np.mean(ls))


def plot(results):
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ps = [r["p_code"] * 100 for r in results]
    ax.plot(ps, [r["web_val"] for r in results], "o-", color=C["blue"], label="网页 val loss")
    ax.plot(ps, [r["code_val"] for r in results], "s-", color=C["orange"], label="代码 val loss")
    for r in results:
        up = r["code_val"] > r["web_val"]                 # 谁在上面，标签就往上放
        ax.annotate(f"{r['web_val']:.2f}", (r["p_code"] * 100, r["web_val"]), (0, -12 if up else 7), textcoords="offset points", ha="center", fontsize=7.5, color=C["blue"])
        ax.annotate(f"{r['code_val']:.2f}", (r["p_code"] * 100, r["code_val"]), (0, 7 if up else -12), textcoords="offset points", ha="center", fontsize=7.5, color=C["orange"])
    ax.set(xlabel="训练 batch 里代码 token 的比例（%）", ylabel="val loss（各自的域）", xticks=ps,
           title="配比是取舍：多给一个域，另一个域就变差——但第一份最值钱")
    ax.legend(fontsize=8)
    save(fig, "e2e-mixture")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(json.load(open(f"{DATA}/mixture.json"))); sys.exit()
    tok = Tokenizer.from_file(f"{DATA}/tokenizer.json")
    web = {"train": np.memmap(f"{DATA}/train.bin", dtype=np.uint16, mode="r"), "val": np.memmap(f"{DATA}/val.bin", dtype=np.uint16, mode="r")}
    code_all = build_code_tokens(tok)
    n_val = len(code_all) // 50
    code = {"train": code_all[:-n_val], "val": code_all[-n_val:]}
    print(f"网页 train {fmt(len(web['train']))} token / val {fmt(len(web['val']))}；代码 train {fmt(len(code['train']))} / val {fmt(len(code['val']))}")
    print(f"每个配比：2 层 × 128 宽，{STEPS} 步 × {BS} × {BLOCK} = {fmt(STEPS * BS * BLOCK)} token，同样的配方；只有 batch 里代码的比例在变\n")
    results = []
    for p in MIXES:
        data = MixData(web, code, p)
        with contextlib.redirect_stdout(io.StringIO()):
            m = make_model(2, 128)
        with contextlib.redirect_stdout(io.StringIO()):
            train(m, data, STEPS, BS, peak_lr=1.4e-3, warmup=max(5, STEPS // 20), log_every=10**9, verbose=False, eval_batches=4)
        lw, lc = domain_loss(m, data, "val"), domain_loss(m, data, "code_val")
        results.append({"p_code": p, "web_val": lw, "code_val": lc})
        print(f"  代码占 {p:4.0%}：网页 val loss {lw:.3f}   代码 val loss {lc:.3f}")
    json.dump(results, open(f"{DATA}/mixture.json", "w"), indent=1)

    r0, r1 = results[0], results[-1]
    print(f"\n读法：代码从 0% 加到 25%，网页 loss {r0['web_val']:.3f} → {results[1]['web_val']:.3f}（+{results[1]['web_val']-r0['web_val']:.3f}），"
          f"代码 loss {r0['code_val']:.3f} → {results[1]['code_val']:.3f}（{results[1]['code_val']-r0['code_val']:+.3f}）——第一份代码最值钱；"
          f"全代码时网页 loss {r1['web_val']:.3f}，比全网页高 {r1['web_val']-r0['web_val']:.2f}。")

    plot(results)
