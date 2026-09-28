"""案例（DL 06）：字符级 LSTM 写莎士比亚（与 Transformer 系列 04 的 nanoGPT 同一份语料、同一预算），以及 seq2seq + attention 的对齐矩阵。
https://arganzheng.life/rnn-lstm-and-the-birth-of-attention.html

    python case_06_char_lstm_and_alignment.py            # 全部：lstm align
    python case_06_char_lstm_and_alignment.py align

lstm   2 层 LSTM，0.87M 参数，batch 12 × 64 字符 × 2000 步（nanoGPT README 小配置的同一预算），val loss 与 nanoGPT 4 层 128 维的 1.66 对照
align  复用 06_rnn_attention.py 的 Seq2Seq（GRU + Bahdanau attention）做"倒序"任务，画出学到的对齐矩阵
输出 out/case-06-*.svg。
"""
import importlib
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from _plot import C, plt, save

HERE = os.path.dirname(os.path.abspath(__file__))
SHAKESPEARE = os.path.join(HERE, "..", "transformer-and-llm", "nanogpt", "data", "shakespeare_char", "input.txt")
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


class CharLSTM(nn.Module):
    def __init__(self, V, d=256, layers=2):
        super().__init__()
        self.emb = nn.Embedding(V, d); self.lstm = nn.LSTM(d, d, layers, batch_first=True); self.head = nn.Linear(d, V)

    def forward(self, x, state=None):
        h, state = self.lstm(self.emb(x), state)
        return self.head(h), state


def load_text():
    if not os.path.exists(SHAKESPEARE):
        import urllib.request
        os.makedirs(os.path.dirname(SHAKESPEARE), exist_ok=True)
        urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", SHAKESPEARE)
    text = open(SHAKESPEARE).read()
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(ids))
    return ids[:n], ids[n:], chars


def get_batch(data, B, T, g):
    ix = torch.randint(len(data) - T - 1, (B,), generator=g)
    x = torch.stack([data[i:i + T] for i in ix]); y = torch.stack([data[i + 1:i + T + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model, data, B, T, g, iters=20):
    model.eval(); tot = 0.0
    for _ in range(iters):
        x, y = get_batch(data, B, T, g); logits, _ = model(x); tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
    model.train(); return tot / iters


@torch.no_grad()
def generate(model, chars, prompt="\n", n=400, temperature=0.8):
    stoi = {c: i for i, c in enumerate(chars)}
    x = torch.tensor([[stoi[c] for c in prompt]], device=DEVICE); state = None; out = []
    logits, state = model(x, state)
    for _ in range(n):
        p = (logits[0, -1] / temperature).softmax(-1); nxt = torch.multinomial(p, 1)
        out.append(chars[nxt.item()]); logits, state = model(nxt[None], state)
    return prompt + "".join(out)


def exp_lstm():
    train, val, chars = load_text()
    V = len(chars); B, T, steps = 12, 64, 2000
    torch.manual_seed(0); g = torch.Generator().manual_seed(1)
    model = CharLSTM(V).to(DEVICE); P = sum(p.numel() for p in model.parameters())
    print(f"=== A. 字符级 LSTM 写莎士比亚（{DEVICE}）===")
    print(f"  语料 {len(train) + len(val):,} 字符，词表 {V}；模型 2 层 LSTM × 256，{P / 1e6:.2f}M 参数（nanoGPT 4 层 128 维是 0.80M）")
    print(f"  预算与 nanoGPT README 小配置相同：batch {B} × {T} 字符 × {steps} 步 = {B * T * steps / 1e6:.2f}M 字符（不到语料的 1.5 遍）")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(1, s / steps)))))
    hist = []; t0 = time.time()
    for s in range(steps + 1):
        if s % 250 == 0:
            tr = estimate_loss(model, train, B, T, torch.Generator().manual_seed(2)); va = estimate_loss(model, val, B, T, torch.Generator().manual_seed(3))
            hist.append((s, tr, va)); print(f"  step {s:<5} train loss {tr:.4f}, val loss {va:.4f}")
        if s == steps:
            break
        x, y = get_batch(train, B, T, g); logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    print(f"  {steps} 步 {time.time() - t0:.0f}s")
    print(f"  对照 nanoGPT 同预算（Transformer 系列第四篇）：step 0 4.17 → step 2000 val 1.66（train 1.50）。"
          f"LSTM：{hist[0][2]:.2f} → {hist[-1][2]:.2f}（train {hist[-1][1]:.2f}）。")
    print("  解读：同一份数据、相近的参数量、同样多的步数，字符级 LSTM 与 4 层小 Transformer 落在同一量级（1.71 vs 1.66）——1 MB 的莎士比亚上看不出 Transformer 的优势；"
          "LSTM 甚至快得多（27 s vs 7 分钟），因为 nanoGPT 的小 batch 在 MPS 上是 kernel 启动瓶颈，不是算力瓶颈。"
          "差别在两处：（一）LSTM 每步只能看一个字符、状态是 256 个数，上下文再长也只能靠这 256 个数记；Transformer 直接看 64 个字符；"
          "（二）时间：LSTM 的 64 步是顺序的，Transformer 的 64 个位置是并行的一次矩阵乘——第七章的两个致命缺点，在这个规模上还不致命，规模上去就是。")
    print("\n  生成 400 字符（temperature 0.8）：")
    sample = generate(model, chars)
    print("  " + sample.replace("\n", "\n  "))

    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    st = [h[0] for h in hist]
    ax.plot(st, [h[1] for h in hist], "o-", ms=3, c=C["blue"], label="LSTM train")
    ax.plot(st, [h[2] for h in hist], "s-", ms=3, c=C["red"], label="LSTM val")
    ng = [(0, 4.1676, 4.1649), (250, 2.8491, 2.8662), (500, 2.3961, 2.4026), (750, 2.1386, 2.1655), (1000, 1.8985, 1.9509),
          (1250, 1.6965, 1.8532), (1500, 1.5555, 1.7179), (1750, 1.4989, 1.6616), (2000, 1.4977, 1.6599)]
    ax.plot([r[0] for r in ng], [r[2] for r in ng], "^--", ms=3, c=C["gray"], label="nanoGPT 4 层 val（Transformer 04）")
    ax.axhline(np.log(V), ls=":", c=C["gray"], lw=0.8); ax.text(1300, np.log(V) - 0.25, f"ln {V} = {np.log(V):.2f}（65 个字符乱猜）", fontsize=7, color=C["gray"])
    ax.set_xlabel("步（batch 12 × 64 字符）"); ax.set_ylabel("loss（每字符）"); ax.legend(fontsize=7); ax.set_title("同一份莎士比亚、同一预算：LSTM vs 小 Transformer")
    save(fig, "case-06-char-lstm")


def exp_align():
    print("\n=== B. seq2seq「把序列倒过来」：GRU + Bahdanau attention 学到的对齐 ===")
    m06 = importlib.import_module("06_rnn_attention")
    V, d, T, steps = 20, 64, 8, 2000
    torch.manual_seed(0); m = m06.Seq2Seq(V, d, True); opt = torch.optim.Adam(m.parameters(), 2e-3); g = torch.Generator().manual_seed(1)
    for s in range(steps):
        src = torch.randint(0, V, (64, T), generator=g); tgt = src.flip(1)
        tgt_in = torch.cat([torch.full((64, 1), V), tgt[:, :-1]], 1)
        logits, _ = m(src, tgt_in); loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    with torch.no_grad():
        src = torch.randint(0, V, (1000, T), generator=g); tgt = src.flip(1)
        tgt_in = torch.cat([torch.full((1000, 1), V), tgt[:, :-1]], 1); logits, al = m(src, tgt_in)
        seq_acc = (logits.argmax(-1) == tgt).all(1).float().mean().item()
    A = al[0].numpy()
    print(f"  T={T}，{steps} 步后整句正确率 {seq_acc:.1%}；一个例子的对齐矩阵（行 = 输出第 t 步，列 = 输入位置，值 = 注意力权重 α_tj）：")
    for r in A:
        print("   " + " ".join(f"{v:.2f}" for v in r))
    print("  解读：大体一条反对角线——输出第 1 个字符时 89% 的注意力在输入最后一个位置，之后每步往前挪一格。"
          "细看偏了一格：第 2 步还在看 x8，之后第 t 步看的是 x(T+2−t)——解码器在教师强制下已经拿到了上一步的正确输出，注意力学成了'找到我刚输出的那个字符在输入里的位置'，"
          "再靠 GRU 状态往前推一格。没有人告诉模型'倒序'意味着什么、也没有人规定对齐要长成人想的样子，它是从数据里学出来的。"
          "Bahdanau 等 2015 论文里英法翻译的对齐图是同一个东西，只是那条线大体沿对角线、局部有交叉（形容词后置）。")

    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(T)); ax.set_xticklabels([f"x{j + 1}" for j in range(T)]); ax.set_yticks(range(T)); ax.set_yticklabels([f"y{t + 1}" for t in range(T)])
    ax.set_xlabel("输入位置（编码器）"); ax.set_ylabel("输出步（解码器）"); ax.set_title(f"倒序任务学到的对齐 α（整句正确率 {seq_acc:.0%}）")
    for t in range(T):
        for j in range(T):
            if A[t, j] > 0.15:
                ax.text(j, t, f"{A[t, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if A[t, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, "case-06-alignment")


EXPS = {"lstm": exp_lstm, "align": exp_align}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    for n in names:
        EXPS[n]()
