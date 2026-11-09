"""深度学习基础（09）：Embedding 与表示学习 配套玩具与配图。
https://arganzheng.life/embedding-and-representation-learning.html

用法：python case_09_emb_toys.py onehot-lookup|figures

onehot-lookup：V=5、d=3、小整数 W；e_2^T W = W 的第 2 行；
               nn.Embedding 与 F.one_hot @ W 差 0；
               序列 [2,0,2] 的 W.grad 只有第 0、2 行非零、第 2 行 2 倍
"""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BLOG_IMG = "/Users/devin/repos/arganzheng-github-com/img/in-post"
spec = importlib.util.spec_from_file_location(
    "lst_plot", "/Users/devin/repos/ai-learning-labs/large-scale-training/_plot.py")
lst_plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lst_plot)
plt, save, text, box = (lst_plot.plt, lst_plot.save, lst_plot.text,
                        lst_plot.box)
C = lst_plot.C
lst_plot.OUT = BLOG_IMG
plt.rcParams["axes.unicode_minus"] = False


def base_fig(title, w=7.6, h=4.3, xlim=(0, 12), ylim=(0, 7)):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    text(ax, .15, ylim[1] - .3, title, fontsize=10.5, fontweight=600,
         color="#111111")
    return fig, ax


def grid(ax, x, y, vals, cw=.6, ch=.6, fs=10.5, fc="#dce9f7", ec="#7a7a7a",
         zero_grey=False):
    vals = np.asarray(vals)
    for i, row in enumerate(np.atleast_2d(vals)):
        for j, v in enumerate(np.atleast_1d(row)):
            grey = zero_grey and v == 0
            p = box(ax, x + j * cw, y - (i + 1) * ch, cw - .02, ch - .02,
                    fc="#ececec" if grey else fc, ec=ec, lw=.4, r=.005)
            text(ax, x + j * cw + (cw - .02) / 2,
                 y - (i + 1) * ch + (ch - .02) / 2,
                 f"{v:g}", in_box=p, ha="center", va="center", fontsize=fs)


# ---------------------------------------------------------------- toys

W = np.array([[2, 1, 0],
              [0, 3, 1],
              [1, 0, 2],
              [4, 1, 1],
              [0, 2, 3]], dtype=float)   # V=5、d=3，小整数


def cmd_onehot_lookup():
    import torch
    e2 = np.eye(5)[2]
    print("e_2 =", e2.astype(int).tolist())
    print("e_2^T W =", (e2 @ W).tolist(), "  == W 的第 2 行", W[2].tolist())
    tW = torch.tensor(W, dtype=torch.float32)
    emb = torch.nn.Embedding(5, 3)
    emb.weight.data = tW.clone()
    a = emb(torch.tensor(2))
    b = torch.nn.functional.one_hot(torch.tensor(2), 5).float() @ tW
    print(f"nn.Embedding(2) = {a.tolist()}, max diff = "
          f"{(a - b).abs().max().item():.1e}")
    # 序列 [2,0,2] 求和反向：只有被查的行有梯度，重复查的行梯度累加
    emb2 = torch.nn.Embedding(5, 3)
    emb2.weight.data = tW.clone()
    out = emb2(torch.tensor([2, 0, 2])).sum()
    out.backward()
    g = emb2.weight.grad.numpy()
    print("loss = sum(emb[2]+emb[0]+emb[2])，对 W 反向：")
    print("  W.grad 非零行：", np.nonzero(np.abs(g).sum(1))[0].tolist())
    print("  W.grad[0] =", g[0].tolist(), "（1 次）")
    print("  W.grad[2] =", g[2].tolist(), "（2 次累加，每行全为 2.0）")
    print("  其余行全 0 —— 梯度是稀疏的")


# ---------------------------------------------------------------- figures

def figure_onehot_lookup():
    fig, ax = base_fig(r"查表 $=$ one-hot 行向量 $\times$ 表 $=$ 取第 $i$ 行"
                       "；反向只写被查到的行",
                       h=3.9, xlim=(0, 12.6), ylim=(0.6, 7.0))
    # ---- 左：e_2^T × W(5×3) = W 第 2 行
    cw, ch = .62, .62
    text(ax, .3, 6.35, r"$e_2^{\!\top}$（$1\times5$）", fontsize=10.5)
    grid(ax, .3, 6.05, np.eye(5)[2], cw=cw, ch=ch, fs=10.5, zero_grey=True)
    # e_2 的第 2 格高亮
    box(ax, .3 + 2 * cw, 6.05 - ch, cw - .02, ch - .02,
        fc="none", ec=C["orange"], lw=1.6, r=.01)
    text(ax, 3.95, 5.5, r"$\times$", fontsize=14, va="center")
    text(ax, 4.6, 6.35, r"表 $W$（$5\times3$）", fontsize=10.5)
    for i in range(5):
        for j in range(3):
            hi = (i == 2)
            p = box(ax, 4.6 + j * cw, 6.05 - (i + 1) * ch, cw - .02,
                    ch - .02, fc="#fae6d6" if hi else "#dce9f7",
                    ec=C["orange"] if hi else "#7a7a7a",
                    lw=1.4 if hi else .4, r=.005)
            text(ax, 4.6 + j * cw + (cw - .02) / 2,
                 6.05 - (i + 1) * ch + (ch - .02) / 2,
                 f"{W[i, j]:g}", in_box=p, ha="center", va="center",
                 fontsize=10.5)
    box(ax, 4.6, 6.05 - 3 * ch, 3 * cw - .02, ch - .02,
        fc="none", ec=C["orange"], lw=1.8, r=.01)
    text(ax, 7.05, 5.5, r"$=$", fontsize=14, va="center")
    text(ax, 7.75, 6.35, r"第 $2$ 行（$1\times3$）", fontsize=10.5)
    grid(ax, 7.75, 6.05, W[2], cw=cw, ch=ch, fs=10.5, fc="#fae6d6",
         ec=C["orange"])
    text(ax, .3, 2.7,
         r"$e_i^{\!\top}W=W_i$：矩阵乘法只做了一次「取行」，"
         "所以实现时直接按 ID 查表",
         fontsize=10.5, color="#555555")
    # ---- 右下：W.grad 5×3，只有第 0、2 行非零
    gx = 10.05
    text(ax, gx, 6.35, r"梯度 $\nabla_W L$（$5\times3$）", fontsize=10.5)
    g = np.zeros((5, 3))
    g[0] = 1
    g[2] = 2
    for i in range(5):
        for j in range(3):
            v = g[i, j]
            p = box(ax, gx + j * cw, 6.05 - (i + 1) * ch, cw - .02,
                    ch - .02,
                    fc="#fae6d6" if v else "#f0f0f0",
                    ec=C["orange"] if v else "#bbbbbb",
                    lw=1.2 if v else .4, r=.005)
            text(ax, gx + j * cw + (cw - .02) / 2,
                 6.05 - (i + 1) * ch + (ch - .02) / 2,
                 f"{v:g}", in_box=p, ha="center", va="center",
                 fontsize=10.5)
    text(ax, gx, 2.6, "$[2,0,2]$ 求和反向：\n"
         "第 $0$ 行用 $1$ 次（全 $1$）；\n"
         "第 $2$ 行用 $2$ 次（全 $2$）；\n"
         "其余行没查到（全 $0$）",
         fontsize=10.5, color="#555555", va="top")
    lst_plot.save(fig, "emb-onehot-lookup")


def figures():
    figure_onehot_lookup()


COMMANDS = {"onehot-lookup": cmd_onehot_lookup, "figures": figures}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "onehot-lookup"
    COMMANDS[cmd]()
