"""算法工程师的数学（05）：从最大似然到交叉熵——文中全部数字与图。
https://arganzheng.life/from-maximum-likelihood-to-cross-entropy.html

    python 05_mle_to_cross_entropy.py            # 全部，约 1–2 分钟（含 GPT-2 与 Qwen tokenizer 的两段，需本地缓存）
    python 05_mle_to_cross_entropy.py coin lm    # 只跑指定的段

段：coin（硬币的最大似然）· lm（三个 token 的玩具语言模型）· underflow（为什么取对数）
    · nll（−log p 曲线）· softmax（对拍、性质、溢出）· lnv（初始 loss ≈ ln V：nanoGPT 日志 + 三个 bug）
    · sampling（GPT-2 上的温度 / top-k / top-p）· mask（SFT 的 loss mask）· rm（奖励模型的 −log σ）· log（读一条训练日志）
"""
import math
import os
import pickle
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

from _plot import C, plt, save

HERE = os.path.dirname(os.path.abspath(__file__))
TLL = os.path.join(HERE, "..", "transformer-and-llm")
sys.path.insert(0, TLL)
torch.manual_seed(0)
np.random.seed(0)
LN65 = math.log(65)


def section(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


# ---------------------------------------------------------------- coin
def coin():
    section("硬币：抛 10 次 7 次正面，正面概率 p 是多少？")
    n, k = 10, 7
    p = np.linspace(0.001, 0.999, 999)
    L = p ** k * (1 - p) ** (n - k)
    logL = k * np.log(p) + (n - k) * np.log(1 - p)
    print("p 的候选值与「这 10 次结果出现的概率」（似然）：")
    for pv in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
        print(f"  p = {pv:.1f}   L(p) = {pv**k:.5f} × {(1-pv)**(n-k):.5f} = {pv**k*(1-pv)**(n-k):.6f}   log L = {k*math.log(pv)+(n-k)*math.log(1-pv):.3f}")
    print(f"似然最大的 p = {p[L.argmax()]:.3f}，log 似然最大的 p = {p[logL.argmax()]:.3f}（同一个点，= k/n = 0.7）")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    axes[0].plot(p, L, color=C["blue"])
    axes[0].axvline(0.7, color=C["red"], ls="--", lw=1)
    axes[0].set(xlabel="猜的正面概率 p", ylabel="似然 L(p) = p⁷(1−p)³", title="似然：哪个 p 让「7 正 3 反」最可能出现")
    axes[0].annotate("最高点 p = 0.7", (0.7, L.max()), (0.2, L.max() * 0.9), arrowprops=dict(arrowstyle="->", color=C["gray"]), color=C["red"])
    axes[1].plot(p, logL, color=C["orange"])
    axes[1].axvline(0.7, color=C["red"], ls="--", lw=1)
    axes[1].set(xlabel="猜的正面概率 p", ylabel="对数似然 log L(p)", ylim=(-25, 0), title="取对数：形状变了，最高点没变")
    save(fig, "05-coin-likelihood")


# ---------------------------------------------------------------- lm
TOK = ["a", "b", "c"]
# 两个候选模型：p(下一个 | 上一个)，行 = 上一个 token，列 = 下一个 token
THETA_1 = np.array([[0.1, 0.8, 0.1],     # a 后面：多半是 b
                    [0.7, 0.1, 0.2],     # b 后面：多半是 a
                    [0.6, 0.2, 0.2]])    # c 后面：多半是 a
THETA_2 = np.array([[0.4, 0.3, 0.3],
                    [0.3, 0.4, 0.3],
                    [0.3, 0.3, 0.4]])
SENT = "a b a c a b".split()


def lm():
    section("三个 token 的玩具语言模型：训练句子 'a b a c a b'，两个候选 θ 谁的似然大？")
    idx = {t: i for i, t in enumerate(TOK)}
    rows = []
    for name, th in (("θ₁", THETA_1), ("θ₂", THETA_2)):
        ps = [th[idx[SENT[t - 1]], idx[SENT[t]]] for t in range(1, len(SENT))]
        prod = float(np.prod(ps))
        logs = [math.log(x) for x in ps]
        rows.append((name, ps, prod, sum(logs)))
        print(f"{name}: 逐位概率 " + "  ".join(f"p({SENT[t]}|{SENT[t-1]})={ps[t-1]:.2f}" for t in range(1, len(SENT))))
        print(f"     乘积 = {prod:.6f}    log 和 = {sum(logs):.3f}    每 token 平均 −log = {-sum(logs)/len(ps):.3f}")
    print("→ 最大似然选 θ₁：它让训练句子出现的概率大 %.0f 倍。" % (rows[0][2] / rows[1][2]))

    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    ax.set_xlim(-0.5, len(SENT) + 0.6)
    ax.set_ylim(-0.4, 2.6)
    ax.axis("off")
    for t, tok in enumerate(SENT):
        ax.add_patch(plt.Rectangle((t - 0.35, 1.9), 0.7, 0.6, fc="#eef3fa", ec=C["blue"]))
        ax.text(t, 2.2, tok, ha="center", va="center", fontsize=13, fontweight="bold")
        if t > 0:
            ax.annotate("", (t - 0.35, 2.2), (t - 0.65, 2.2), arrowprops=dict(arrowstyle="->", color=C["gray"]))
            for r, (name, ps, prod, ls) in enumerate(rows):
                y = 1.2 - r * 0.7
                ax.text(t, y, f"{ps[t-1]:.2f}", ha="center", va="center", color=[C["green"], C["orange"]][r], fontsize=11)
    ax.text(-0.45, 1.2, "θ₁", ha="right", va="center", color=C["green"], fontsize=11, fontweight="bold")
    ax.text(-0.45, 0.5, "θ₂", ha="right", va="center", color=C["orange"], fontsize=11, fontweight="bold")
    ax.text(0, 1.2, "—", ha="center", va="center", color=C["gray"])
    ax.text(0, 0.5, "—", ha="center", va="center", color=C["gray"])
    for r, (name, ps, prod, ls) in enumerate(rows):
        ax.text(len(SENT) - 0.4, 1.2 - r * 0.7, f"×→ {prod:.5f}\nlog 和 {ls:.2f}", va="center", fontsize=9, color=[C["green"], C["orange"]][r])
    ax.text(len(SENT) / 2 - 0.5, -0.2, "每个位置：模型给「真实的下一个 token」的概率；整句的概率 = 逐位相乘（链式法则）", ha="center", fontsize=9, color=C["gray"])
    save(fig, "05-toy-lm-likelihood")


# ---------------------------------------------------------------- underflow
def underflow():
    section("为什么取对数：1000 个 0.01 相乘")
    prod = 1.0
    for _ in range(1000):
        prod *= 0.01
    print(f"  直接相乘（float64）：{prod!r}   ← 已经是 0，信息全丢")
    print(f"  log 求和：1000 × log(0.01) = {1000 * math.log(0.01):.1f}   ← 正常的数")
    print(f"  float64 能表示的最小正数约 {np.finfo(np.float64).tiny:.1e}，0.01^1000 = 1e-2000 远小于它")
    ps = np.array([0.9, 0.5, 0.1, 0.01, 0.001])
    print("  单调性：p 越大，log p 越大：", " ".join(f"{p}→{math.log(p):.2f}" for p in ps))


# ---------------------------------------------------------------- nll
def nll():
    section("−log p：模型给真实 token 的概率 p 越小，惩罚越大")
    for p in (0.99, 0.9, 0.5, 0.1, 0.01, 0.001):
        print(f"  p = {p:<6} −ln p = {-math.log(p):6.3f} nat = {-math.log2(p):6.3f} bit")
    p = np.linspace(0.005, 1, 400)
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    ax.plot(p, -np.log(p), color=C["blue"], lw=2)
    for pv, txt in ((0.9, "猜对了 9 成把握\n惩罚 0.11"), (0.5, "对半开\n惩罚 0.69"), (0.1, "只给了 10%\n惩罚 2.3"), (0.01, "只给了 1%\n惩罚 4.6")):
        ax.plot(pv, -math.log(pv), "o", color=C["red"])
        ax.annotate(txt, (pv, -math.log(pv)), (pv + 0.06, -math.log(pv) + 0.4), fontsize=8, arrowprops=dict(arrowstyle="-", color=C["gray"], lw=0.6))
    ax.set(xlabel="模型给真实 token 的概率 p", ylabel="loss = −ln p（nat）", ylim=(0, 5.5), title="每个位置的 loss：p → 1 时是 0，p → 0 时冲向无穷")
    save(fig, "05-neg-log-curve")


# ---------------------------------------------------------------- softmax
def softmax():
    section("softmax：手算 vs PyTorch；三条性质；溢出")
    z = torch.tensor([2.0, 1.0, 0.0])
    e = torch.exp(z)
    p = e / e.sum()
    print(f"  logits {z.tolist()} → e^z {[round(x, 3) for x in e.tolist()]} 和 {e.sum():.3f} → p {[round(x, 3) for x in p.tolist()]}")
    print(f"  torch.softmax 给的：{[round(x, 3) for x in torch.softmax(z, 0).tolist()]}")
    y = torch.tensor(1)
    print(f"  真实 token 是第 2 个：手算 −log p₂ = {-math.log(p[1]):.4f}；F.cross_entropy(z, y) = {F.cross_entropy(z[None], y[None]):.4f}")
    print(f"  保序：p 的排序 {torch.argsort(p, descending=True).tolist()} = logits 的排序 {torch.argsort(z, descending=True).tolist()}")
    print(f"  差值决定比值：p₁/p₂ = {p[0]/p[1]:.3f} = e^(2−1) = {math.e:.3f}；p₁/p₃ = {p[0]/p[2]:.3f} = e² = {math.e**2:.3f}")
    print(f"  平移不变：softmax(z + 100) = {[round(x, 3) for x in torch.softmax(z + 100, 0).tolist()]}")
    big = torch.tensor([1000.0, 1001.0, 999.0])
    naive = torch.exp(big) / torch.exp(big).sum()
    print(f"  溢出：直接算 e^{big.tolist()} = {torch.exp(big).tolist()} → p = {naive.tolist()}")
    m = big.max()
    stable = torch.exp(big - m) / torch.exp(big - m).sum()
    print(f"  先减最大值 {m:.0f}：e^(z−m) = {[round(x, 3) for x in torch.exp(big - m).tolist()]} → p = {[round(x, 3) for x in stable.tolist()]}")

    x = np.linspace(-3, 3, 200)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    axes[0].plot(x, np.exp(x), color=C["blue"])
    for zv in (0, 1, 2):
        axes[0].plot(zv, math.exp(zv), "o", color=C["red"])
        axes[0].annotate(f"e^{zv} = {math.exp(zv):.2f}", (zv, math.exp(zv)), (zv - 2.4, math.exp(zv) + 1.5), fontsize=8, arrowprops=dict(arrowstyle="-", color=C["gray"], lw=0.6))
    axes[0].set(xlabel="logit z", ylabel="e^z", title="指数：永远是正数，差 1 就放大 2.7 倍")
    taus = [0.5, 1, 2]
    for i, tau in enumerate(taus):
        pt = torch.softmax(z / tau, 0).numpy()
        axes[1].bar(np.arange(3) + (i - 1) * 0.26, pt, 0.24, label=f"τ = {tau}", color=[C["red"], C["blue"], C["green"]][i])
    axes[1].set_xticks(range(3), ["token 1\n(z=2)", "token 2\n(z=1)", "token 3\n(z=0)"])
    axes[1].set(ylabel="概率", title="同一组 logits，温度改变「多确定」")
    axes[1].legend()
    save(fig, "05-exp-and-temperature")


# ---------------------------------------------------------------- lnv
def _shakespeare_batch(bs, block):
    data = np.memmap(os.path.join(TLL, "nanogpt", "data", "shakespeare_char", "train.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block - 1, (bs,))
    x = torch.stack([torch.from_numpy(data[i:i + block].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block].astype(np.int64)) for i in ix])
    return x, y


def _tiny_gpt(vocab=65):
    from nanogpt_model import GPT, GPTConfig
    torch.manual_seed(0)
    return GPT(GPTConfig(block_size=64, vocab_size=vocab, n_layer=2, n_head=4, n_embd=64, dropout=0.0, bias=False))


def lnv():
    section("初始 loss ≈ ln V：nanoGPT 的真实日志 + 三个 bug 各长什么样")
    meta = pickle.load(open(os.path.join(TLL, "nanogpt", "data", "shakespeare_char", "meta.pkl"), "rb"))
    print(f"  莎士比亚字符级：V = {meta['vocab_size']}，ln V = {LN65:.4f}")
    log = open(os.path.join(TLL, "expected", "train_shakespeare_char_base.txt")).read()
    steps, tr, va = [], [], []
    for m in re.finditer(r"step (\d+): train loss ([\d.]+), val loss ([\d.]+)", log):
        steps.append(int(m.group(1))); tr.append(float(m.group(2))); va.append(float(m.group(3)))
    print(f"  Transformer 04 那次训练（0.80M 参数）的日志：step 0 train loss {tr[0]}，val {va[0]}；最后 step {steps[-1]} val {va[-1]}")

    # 四种情况各训 150 步：正常 / 输出层初始化放大 / 目标泄漏进输入 / padding 算进 loss
    variants = {}
    for name in ("正常", "输出层权重 ×20", "泄漏：输入 = 目标", "一半目标是 pad 且算进 loss"):
        torch.manual_seed(0)
        vocab = 66 if "pad" in name else 65
        model = _tiny_gpt(vocab)
        if "×20" in name:
            with torch.no_grad():
                model.lm_head.weight.mul_(20)          # tied 到 embedding，一起放大
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = []
        for step in range(150):
            x, y = _shakespeare_batch(32, 64)
            if "泄漏" in name:
                x = y.clone()                            # 忘了错一位：模型看着答案预测答案
            if "pad" in name:
                y = y.clone(); y[:, ::2] = 65            # 一半位置的目标是 pad，却没有 ignore_index
            logits, _ = model(x, y)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(loss.item())
            opt.zero_grad(); loss.backward(); opt.step()
        variants[name] = losses
        print(f"  {name:<22} step 0 loss {losses[0]:6.3f}   step 50 {losses[50]:6.3f}   step 149 {losses[-1]:6.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    axes[0].plot(steps, tr, label="train", color=C["blue"]); axes[0].plot(steps, va, label="val", color=C["orange"])
    axes[0].axhline(LN65, color=C["red"], ls="--", lw=1, label=f"ln 65 = {LN65:.2f}")
    axes[0].annotate(f"step 0: {tr[0]}", (0, tr[0]), (500, 3.9), arrowprops=dict(arrowstyle="->", color=C["gray"]), fontsize=8)
    axes[0].set(xlabel="step", ylabel="loss（nat/字符）", title="nanoGPT 莎士比亚：从 ln V 开始往下走")
    axes[0].legend()
    for (name, ls), col in zip(variants.items(), [C["blue"], C["purple"], C["red"], C["orange"]]):
        axes[1].plot(ls, label=name, color=col, lw=1.4)
    axes[1].axhline(LN65, color=C["gray"], ls="--", lw=1)
    axes[1].text(148, LN65 + 0.15, "ln V", ha="right", fontsize=8, color=C["gray"])
    axes[1].set(xlabel="step", ylabel="loss", title="同一个小模型，三种 bug 各长什么样", ylim=(0, 9))
    axes[1].legend(fontsize=7)
    save(fig, "05-lnv-and-bugs")


# ---------------------------------------------------------------- sampling
def sampling():
    section("GPT-2 上的温度 / top-k / top-p：'The capital of France is' 的下一个 token")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        z = model(ids).logits[0, -1]
    print(f"  logits：共 {z.numel()} 个，最大 {z.max():.2f}，最小 {z.min():.2f}，最大与第二大相差 {(z.topk(2).values[0]-z.topk(2).values[1]):.2f}")
    top = torch.softmax(z, 0).topk(10)
    names = [tok.decode([i]).replace(" ", "␣") for i in top.indices]
    print("  τ=1 的前十名：", "  ".join(f"{n}:{p:.3f}" for n, p in zip(names, top.values.tolist())))
    print(f"  前十名合计 {top.values.sum():.3f}；其余 {z.numel()-10} 个 token 合计 {1-top.values.sum():.3f}")
    table = {}
    for tau in (0.5, 1.0, 2.0):
        pt = torch.softmax(z / tau, 0)
        table[tau] = pt[top.indices]
        print(f"  τ={tau}: " + "  ".join(f"{n}:{p:.3f}" for n, p in zip(names, table[tau].tolist())) + f"   前十合计 {pt[top.indices].sum():.3f}")
    p1 = torch.softmax(z, 0)
    srt, _ = p1.sort(descending=True)
    cum = srt.cumsum(0)
    k90 = int((cum < 0.9).sum()) + 1
    print(f"  top-p 0.9 保留 {k90} 个 token；top-k 5 保留 5 个（合计概率 {srt[:5].sum():.3f}）")
    prompt2 = "The United States of"
    with torch.no_grad():
        z2 = model(tok(prompt2, return_tensors="pt").input_ids).logits[0, -1]
    p2 = torch.softmax(z2, 0)
    srt2, _ = p2.sort(descending=True)
    cum2 = srt2.cumsum(0)
    k90_2 = int((cum2 < 0.9).sum()) + 1
    print(f"  对比一个「很确定」的位置：'{prompt2}' → 第一名 {tok.decode([int(p2.argmax())])!r} 概率 {p2.max():.3f}，top-p 0.9 只保留 {k90_2} 个 token")
    torch.manual_seed(0)
    for tau in (0.0, 0.7, 1.5):
        outs = []
        for _ in range(3):
            g = model.generate(ids, max_new_tokens=6, do_sample=tau > 0, temperature=tau if tau > 0 else None, top_k=0 if tau > 0 else None, pad_token_id=tok.eos_token_id)
            outs.append(tok.decode(g[0, ids.size(1):]).replace("\n", "\\n"))
        print(f"  τ={tau} 三次续写：" + " | ".join(repr(o) for o in outs))

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), width_ratios=[3, 2])
    xs = np.arange(10)
    for i, (tau, col) in enumerate(zip((0.5, 1.0, 2.0), (C["red"], C["blue"], C["green"]))):
        axes[0].bar(xs + (i - 1) * 0.27, table[tau].numpy(), 0.25, label=f"τ = {tau}", color=col)
    axes[0].set_xticks(xs, names, rotation=45, ha="right", fontsize=8)
    axes[0].set(ylabel="概率", title="GPT-2：'The capital of France is' 之后的前十个 token")
    axes[0].legend()
    n = np.arange(1, cum.numel() + 1)
    axes[1].semilogx(n, cum.numpy(), color=C["blue"], lw=1.5, label=f"'{prompt}'（开放）")
    axes[1].semilogx(n, cum2.numpy(), color=C["green"], lw=1.5, label=f"'{prompt2}'（确定）")
    axes[1].axhline(0.9, color=C["orange"], ls="--", lw=1)
    axes[1].axvline(k90, color=C["blue"], ls=":", lw=1)
    axes[1].axvline(k90_2, color=C["green"], ls=":", lw=1)
    axes[1].axvline(5, color=C["purple"], ls=":", lw=1)
    axes[1].text(k90 * 1.2, 0.45, f"top-p 0.9\n留 {k90} 个", color=C["blue"], fontsize=8)
    axes[1].text(k90_2 * 1.3, 0.6, f"留 {k90_2} 个", color=C["green"], fontsize=8)
    axes[1].text(5.5, 0.3, "top-k 5", color=C["purple"], fontsize=8)
    axes[1].set(xlabel="按概率排序的 token 名次（对数刻度）", ylabel="累计概率", title="截断：top-k 数个数，top-p 数概率", ylim=(0, 1.02))
    axes[1].legend(fontsize=7, loc="lower right")
    save(fig, "05-gpt2-sampling")


# ---------------------------------------------------------------- mask
def mask():
    section("SFT 的 loss mask：一条对话在 Qwen2.5 chat 模板下哪些 token 算 loss")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    msgs = [{"role": "user", "content": "1+1=?"}, {"role": "assistant", "content": "2"}]
    prompt_ids = tok.apply_chat_template(msgs[:1], add_generation_prompt=True, tokenize=True)
    full_ids = tok.apply_chat_template(msgs, tokenize=True)
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids, full_ids = prompt_ids["input_ids"], full_ids["input_ids"]
    m = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
    toks = [tok.decode([i]).replace("\n", "\\n") for i in full_ids]
    print(f"  共 {len(full_ids)} 个 token，其中 prompt {len(prompt_ids)} 个（mask=0，不算 loss），回答 {sum(m)} 个（mask=1）")
    for t, mm in zip(toks, m):
        print(f"    {mm}  {t!r}")
    print("  loss = 只对 mask=1 的位置求 −log p 再除以 %d，不是除以 %d" % (sum(m), len(full_ids)))
    fig, ax = plt.subplots(figsize=(7.6, 1.9))
    ax.axis("off")
    per_row = 9
    for i, (t, mm) in enumerate(zip(toks, m)):
        r, c = divmod(i, per_row)
        ax.add_patch(plt.Rectangle((c, -r * 1.3), 0.96, 1.0, fc="#fde5e5" if mm else "#eef3fa", ec=C["red"] if mm else C["blue"], lw=0.8))
        ax.text(c + 0.475, -r * 1.3 + 0.62, t if len(t) < 13 else t[:11] + "…", ha="center", va="center", fontsize=7)
        ax.text(c + 0.475, -r * 1.3 + 0.2, str(mm), ha="center", va="center", fontsize=8, color=C["red"] if mm else C["blue"], fontweight="bold")
    rows = (len(toks) - 1) // per_row + 1
    ax.set_xlim(-0.1, per_row)
    ax.set_ylim(-(rows - 1) * 1.3 - 0.5, 1.2)
    ax.text(0, 1.1, "蓝 = prompt（mask 0，不算 loss）   红 = 回答（mask 1，算 loss）", fontsize=8, color=C["gray"])
    save(fig, "05-sft-loss-mask")


# ---------------------------------------------------------------- rm
def rm():
    section("奖励模型的 loss：−log σ(r_w − r_l)")
    d = np.linspace(-6, 6, 300)
    loss = -np.log(1 / (1 + np.exp(-d)))
    for dv in (-2, 0, 2, 4):
        print(f"  r_w − r_l = {dv:+d}   P(好的 ≻ 差的) = σ = {1/(1+math.exp(-dv)):.3f}   loss = {-math.log(1/(1+math.exp(-dv))):.3f}")
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    ax.plot(d, loss, color=C["blue"], lw=2)
    ax.axvline(0, color=C["gray"], ls=":", lw=1)
    ax.text(0.2, 3.5, "分不出好坏：loss = ln 2 = 0.69", fontsize=8)
    ax.text(2.5, 1.2, "好的分高 →\n概率接近 1，loss → 0", fontsize=8, color=C["green"])
    ax.text(-5.8, 3.5, "← 把差的排在前面\nloss 线性增长，被重罚", fontsize=8, color=C["red"])
    ax.set(xlabel="r(好回答) − r(差回答)", ylabel="loss = −log σ(差值)", title="同一个模板：把「A 比 B 好」的概率放进 −log")
    save(fig, "05-reward-model-loss")


# ---------------------------------------------------------------- log
def log():
    section("读一条训练日志")
    text = open(os.path.join(TLL, "expected", "train_shakespeare_char_base.txt")).read()
    lines = [l for l in text.splitlines() if l.startswith("step")]
    print("  原始日志（nanoGPT，字符级莎士比亚，V = 65）：")
    for l in lines[:3] + ["  ..."] + lines[-2:]:
        print("   ", l)
    print(f"\n  {'step':>5} {'val loss':>9} {'PPL=e^loss':>11} {'bits/char':>10}  含义")
    for l in lines[:1] + lines[3:4] + lines[-1:]:
        m = re.match(r"step (\d+): train loss ([\d.]+), val loss ([\d.]+)", l)
        s, trl, val = int(m.group(1)), float(m.group(2)), float(m.group(3))
        note = {0: f"≈ ln 65 = {LN65:.2f}：均匀乱猜，65 选 1", }.get(s, "")
        if s == int(re.match(r"step (\d+)", lines[-1]).group(1)):
            note = f"train {trl} < val {val}：开始轻微过拟合（差 {val-trl:.2f}）"
        print(f"  {s:>5} {val:>9.4f} {math.exp(val):>11.1f} {val/math.log(2):>10.2f}  {note}")
    print("  PPL 读法：模型在每个位置「像是在 e^loss 个等可能的字符里猜」；从 65 猜到约 5 个。")


ALL = {"coin": coin, "lm": lm, "underflow": underflow, "nll": nll, "softmax": softmax, "lnv": lnv,
       "sampling": sampling, "mask": mask, "rm": rm, "log": log}

if __name__ == "__main__":
    picks = [a for a in sys.argv[1:] if a in ALL] or list(ALL)
    for k in picks:
        ALL[k]()
