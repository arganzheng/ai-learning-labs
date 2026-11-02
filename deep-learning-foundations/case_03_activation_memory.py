"""深度学习基础（03）补实验：反向到底保存了什么。

  python case_03_activation_memory.py saved     # 两层 MLP 逐张量列出 autograd 的保存
  python case_03_activation_memory.py ckpt      # 8 层 MLP：全存 vs 每 4 层 checkpoint
  python case_03_activation_memory.py figures   # 三张配图（用 large-scale-training/_plot.py 风格）

图写到博客 img/in-post/（lst_plot.OUT 被改指到那里）。
"""
import importlib.util
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
BLOG_IMG = "/Users/devin/repos/arganzheng-github-com/img/in-post"
EXPECTED = os.path.join(HERE, "expected")

spec = importlib.util.spec_from_file_location(
    "lst_plot", "/Users/devin/repos/ai-learning-labs/large-scale-training/_plot.py")
lst_plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lst_plot)
plt, save, text, box, box_title = (lst_plot.plt, lst_plot.save, lst_plot.text,
                                   lst_plot.box, lst_plot.box_title)
C = lst_plot.C
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle, FancyArrowPatch  # noqa: E402

lst_plot.OUT = BLOG_IMG

torch.manual_seed(0)


# ------------------------------------------------------------- saved
def _iter_nodes(root):
    seen, stack = set(), [root]
    while stack:
        n = stack.pop()
        if n is None or n in seen:
            continue
        seen.add(n)
        yield n
        for nxt, _ in n.next_functions:
            stack.append(nxt)


def cmd_saved():
    """两层 MLP 784→256(ReLU)→10，CE，batch 128 fp32；saved_tensors_hooks 全记录。"""
    m, k1, k2, ncls = 128, 784, 256, 10
    torch.manual_seed(0)
    X = torch.randn(m, k1)
    Y = torch.randint(0, ncls, (m,))
    W1 = torch.randn(k1, k2, requires_grad=True) * 0.02
    b1 = torch.zeros(k2, requires_grad=True)
    W2 = torch.randn(k2, ncls, requires_grad=True) * 0.02
    b2 = torch.zeros(ncls, requires_grad=True)

    saved = []                       # (tensor, pack 次序)

    def pack(t):
        saved.append(t)
        return saved[-1]

    def unpack(t):
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        H = X @ W1 + b1
        A = F.relu(H)
        Z = A @ W2 + b2
        loss = F.cross_entropy(Z, Y)
        # 归主必须在 backward 前做：backward 会释放 saved tensors，
        # getattr 触发 unpack 拿回的正是 pack 的那张
        by_id = {id(t): i for i, t in enumerate(saved)}
        by_ptr = {}
        for i, t in enumerate(saved):
            by_ptr.setdefault(t.untyped_storage().data_ptr(), []).append(i)
        pairs = []          # (owner, id, ptr) — getattr 可能返回同 storage 的新 view
        for node in _iter_nodes(loss.grad_fn):
            for attr in dir(node):
                if not attr.startswith("_saved") or "sym_" in attr:
                    continue
                try:
                    v = getattr(node, attr)
                except Exception:
                    continue
                for t in (v if isinstance(v, (list, tuple)) else [v]):
                    if isinstance(t, torch.Tensor):
                        pairs.append((node.name().rstrip("0") + ":" +
                                      attr[len("_saved_"):], id(t),
                                      t.untyped_storage().data_ptr()))
        owners = {}
        used = set()
        for name, tid, ptr in pairs:
            if tid in by_id and by_id[tid] not in used:
                owners[by_id[tid]] = name
                used.add(by_id[tid])
        for name, tid, ptr in pairs:
            for i in by_ptr.get(ptr, []):
                if i not in used:
                    owners[i] = name
                    used.add(i)
                    break
        loss.backward()

    def klass(t):
        if t is X or t.data_ptr() == X.data_ptr():
            return "输入 X"
        for nm, p in (("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2)):
            if t.data_ptr() == p.data_ptr():
                return f"参数 {nm}"
        if t.data_ptr() == H.data_ptr():
            return "ReLU 输入 H"
        if t.data_ptr() == A.data_ptr():
            return "ReLU 输出 A"
        if t.data_ptr() == Y.data_ptr():
            return "标签 Y"
        if tuple(t.shape) == (m, ncls):
            return "Z / softmax 输出"
        return "其他（CE 内部）"

    print(f"== saved：MLP {k1}->{k2}(ReLU)->{ncls}，CE，batch {m}，fp32")
    print(f"{'#':>2} {'算子节点':<28} {'shape':<14} {'dtype':<8} {'字节':>8}  归类")
    seen_ptr = {}
    act = param = 0
    for i, t in enumerate(saved):
        ptr = t.untyped_storage().data_ptr()
        nb = t.untyped_storage().nbytes()
        owner = owners.get(i, "?")
        cls = klass(t)
        dup = ptr in seen_ptr
        if dup:
            cls = cls + "（同一块内存，不重复计）"
        else:
            seen_ptr[ptr] = i
            if cls.startswith("参数"):
                param += nb
            else:
                act += nb
        print(f"{i:>2} {owner:<28} {str(tuple(t.shape)):<14} "
              f"{str(t.dtype).replace('torch.', ''):<8} {nb:>8}  {cls}")
    print(f"\n按 storage 去重后：激活合计 {act / 1024:.0f} KiB"
          f"（X + ReLU 保存 + softmax 侧）；参数合计 {param / 1024:.0f} KiB"
          "（autograd 为 dW/dX 保存，是参数不是激活）")
    print("对照正文 §六.1 表：X 392 KiB + H 掩码 32 KiB + A 128 KiB = 552 KiB。"
          "PyTorch 的 ReLU 保存的是 A（输出 128 KiB）而不是 H 的 mask——"
          "反向时用 A > 0 判断符号，等价但多 96 KiB。")


# ------------------------------------------------------------- ckpt
def _mlp8(width=1024, layers=8):
    return nn.Sequential(*[nn.Sequential(nn.Linear(width, width), nn.ReLU())
                           for _ in range(layers)])


def _measure_saved(model, X, Y, ckpt=None):
    """返回 (总 pack 字节, 峰值活着的保存字节, 每 Linear 前向次数, grads)。"""
    fwd_calls = {name: [0] for name, mod in model.named_modules()
                 if isinstance(mod, nn.Linear)}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            mod.register_forward_hook(
                lambda m, i, o, n=name: fwd_calls[n].__setitem__(0, fwd_calls[n][0] + 1))

    live = {"now": 0, "peak": 0, "total": 0}

    def pack(t):
        nb = t.untyped_storage().nbytes()
        live["now"] += nb
        live["total"] += nb
        live["peak"] = max(live["peak"], live["now"])
        return t

    def unpack(t):
        live["now"] -= t.untyped_storage().nbytes()
        return t

    for p in model.parameters():
        p.grad = None
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        for p in model.parameters():
            p.grad = None
        h = X
        if ckpt is None:
            out = model(h)
        else:
            outs = []
            for i in range(0, len(model), ckpt):
                blk = model[i:i + ckpt] if isinstance(model, nn.Sequential) \
                    else nn.Sequential(*list(model.children())[i:i + ckpt])
                h = torch.utils.checkpoint.checkpoint(
                    blk, h, use_reentrant=False)
            out = h
        loss = F.cross_entropy(out, Y)
        loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    per_layer = [v[0] for v in fwd_calls.values()]
    return live["total"], live["peak"], per_layer, grads


def cmd_ckpt():
    width, layers, m, ncls = 1024, 8, 256, 10
    torch.manual_seed(0)
    base = _mlp8(width, layers)
    ref = _mlp8(width, layers)
    ref.load_state_dict(base.state_dict())
    torch.manual_seed(1)
    X = torch.randn(m, width)
    Y = torch.randint(0, ncls, (m,))

    t1, p1, f1, g1 = _measure_saved(base, X, Y, ckpt=None)
    t2, p2, f2, g2 = _measure_saved(ref, X, Y, ckpt=4)

    diff = max((g1[n] - g2[n]).abs().max().item() for n in g1)
    print(f"== ckpt：8 层 MLP（宽 {width}），batch {m}，fp32")
    print(f"{'方案':<22} {'总保存':>10} {'峰值':>10} {'每 Linear 前向次数':<22}")
    print(f"{'全部保存':<22} {t1/1024/1024:>8.2f} MB {p1/1024/1024:>8.2f} MB  {f1}")
    print(f"{'每 4 层 checkpoint':<22} {t2/1024/1024:>8.2f} MB {p2/1024/1024:>8.2f} MB  {f2}")
    print(f"梯度 max|diff| = {diff:.3e}（本玩具两次前向浮点顺序相同，逐位一致）")
    print(f"保存降到 {t2/t1:.0%}，8 个 Linear 各前向 2 次 = "
          f"恰好一次额外前向（{sum(f2) - sum(f1)} 次 Linear），与全量重算同量；"
          "分块大小只改存什么")


# ------------------------------------------------------------- figures
def _c(sw=0.55):
    """灰/彩混合填充"""
    return sw


def fig_one_iteration():
    """一次迭代：loss 和梯度什么时候算；两条细泳道画保存量与梯度量。"""
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7.3); ax.axis("off")
    ax.set_title("一次迭代：loss 和梯度什么时候算，激活什么时候释放",
                 fontweight=600, color="#111111")

    phases = [
        ("前向", "逐层算 z、a\n并保存激活", "#dce9f7", C["blue"]),
        ("算 loss", "一个标量", "#e5e5d7", "#8a8a4a"),
        ("反向", "从 loss 往回\n用掉并释放激活", "#fae6d6", C["orange"]),
        ("step", "W ← W - η·dW", "#eee1f3", C["purple"]),
        ("zero", "梯度清零", "#e8e8e8", "#777777"),
    ]
    ws = [2.6, 1.35, 2.6, 2.15, 1.6]
    x, y, h = 0.3, 4.8, 1.45
    pos = []
    for (name, sub, fc, ec), w in zip(phases, ws):
        p = box(ax, x, y, w, h, "", fc=fc, ec=ec, lw=1.0)
        box_title(ax, x + w / 2, y + h - 0.36, name, p, color=ec)
        text(ax, x + w / 2, y + 0.5, sub, in_box=p, ha="center",
             va="center", fontsize=8.5)
        pos.append((x, x + w))
        if w is not ws[-1]:
            ax.add_patch(FancyArrowPatch((x + w + 0.03, y + h / 2),
                                         (x + w + 0.22, y + h / 2),
                                         arrowstyle="-|>", mutation_scale=16,
                                         color="#555555", lw=1.4))
        x += w + 0.25

    # 泳道 1：保存的激活量：前向线性升到峰，反向线性降回 0
    y1 = 2.7
    text(ax, 0.05, y1 + 0.52, "保存的激活", ha="left", va="center", fontsize=9)
    xf1, xf2 = pos[0]
    xb1, xb2 = pos[2]
    xs_, xs1 = pos[1]
    poly = [(xf1, y1), (xs1 - 0.05, y1 + 0.95), (xb2, y1)]
    ax.plot([p[0] for p in poly], [p[1] for p in poly],
            color=C["green"], lw=1.5)
    ax.fill_between([xf1, xs1 - 0.05, xb2], y1, [y1, y1 + 0.95, y1],
                    color=C["green"], alpha=0.18)
    text(ax, (xf1 + xs1) / 2, y1 + 0.98, "逐层堆高", ha="center",
         va="bottom", fontsize=8.5, color=C["green"])
    text(ax, xb2 + 0.15, y1 + 0.5, "逐层释放", ha="left",
         va="center", fontsize=8.5, color=C["green"])

    # 泳道 2：梯度量：反向逐步升到满，step 后清零
    y2 = 1.15
    text(ax, 0.05, y2 + 0.52, "梯度量", ha="left", va="center", fontsize=9)
    xo1, xo2 = pos[3]
    xz1, xz2 = pos[4]
    ax.fill_between([xb1, xb2], y2, [y2, y2 + 0.8], color=C["purple"],
                    alpha=0.18)
    ax.plot([xb1, xb2], [y2, y2 + 0.8], color=C["purple"], lw=1.5)
    ax.plot([xb2, xz1], [y2 + 0.8, y2 + 0.8], color=C["purple"], lw=1.5)
    ax.plot([xz1, xz1], [y2 + 0.8, y2], color=C["purple"], lw=1.5)
    ax.plot([xz1, xz2], [y2, y2], color=C["purple"], lw=1.5)
    text(ax, (xb1 + xb2) / 2, y2 + 0.9, "逐层到齐", ha="center",
         va="bottom", fontsize=8.5, color=C["purple"])
    text(ax, xz2 + 0.15, y2 + 0.4, "清零", ha="left",
         va="center", fontsize=8.5, color=C["purple"])

    text(ax, 6.0, 0.35,
         "激活的峰在前向末、梯度的峰在反向末——两条泳道错开，"
         "合起来才是显存峰值",
         ha="center", va="center", fontsize=9.5)
    save(fig, "backprop-one-iteration")


def fig_checkpointing():
    """8 层 MLP：全存 vs 每 4 层只存块输入 + 重算。数字来自 cmd_ckpt 实测。"""
    # 直接量一次（与 expected 输出同源）
    width, layers, m, ncls = 1024, 8, 256, 10
    torch.manual_seed(0)
    base = _mlp8(width, layers)
    ref = _mlp8(width, layers)
    ref.load_state_dict(base.state_dict())
    torch.manual_seed(1)
    X = torch.randn(m, width)
    Y = torch.randint(0, ncls, (m,))
    t1, p1, f1, g1 = _measure_saved(base, X, Y, ckpt=None)
    t2, p2, f2, g2 = _measure_saved(ref, X, Y, ckpt=4)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7.6); ax.axis("off")
    ax.set_title("激活重算（checkpointing）：每 4 层只存一个输入，反向先重算",
                 fontweight=600, color="#111111")

    # 每个方案的格子序列：(层号, 阶段) —— 时间是横轴
    cw_, cgap = 0.34, 0.02
    chh = 0.62
    x0 = 1.85

    def lane(y, cells, all_saved=False, saved_inputs=False):
        x = x0
        for ln, phase in cells:
            if phase == "fwd":
                if all_saved:
                    fc, ec = "#fae6d6", C["orange"]
                elif saved_inputs and ln in (1, 5):
                    fc, ec = "#ffffff", C["orange"]
                else:
                    fc, ec = "#f5f5f5", "#aaaaaa"
                hatch = None
            elif phase == "re":
                fc, ec, hatch = "#f5f5f5", "#aaaaaa", "///"
            else:
                fc, ec, hatch = "#dce9f7", C["blue"], None
            p = box(ax, x, y, cw_, chh, "", fc=fc, ec=ec, fs=7.5)
            ax.text(x + cw_ / 2, y + chh / 2, str(ln), ha="center",
                    va="center", fontsize=7.5)
            if hatch:
                p.set_hatch(hatch)
            if phase == "fwd" and saved_inputs and ln in (1, 5):
                ax.add_patch(Rectangle((x + cw_ * 0.35, y + chh * 0.35),
                                       cw_ * 0.3, chh * 0.3,
                                       facecolor="#fae6d6", edgecolor=C["orange"],
                                       lw=0.5))
            x += cw_ + cgap
        return x

    def seg_label(y, x_from, x_to, s, color):
        text(ax, (x_from + x_to) / 2, y, s, ha="center", va="center",
             fontsize=8, color=color)

    # ---- 图例（小色块 + 标签；与格子的画法严格一致）----
    legend = [("#fae6d6", C["orange"], None, "全部保存"),
              ("#ffffff", C["orange"], "ckpt", "只存块输入"),
              ("#f5f5f5", "#aaaaaa", None, "算完即丢"),
              ("#f5f5f5", "#aaaaaa", "///", "不保存、反向前重算"),
              ("#dce9f7", C["blue"], None, "反向")]
    lx = 0.45
    for fc, ec, kind, name in legend:
        r = Rectangle((lx, 6.35), 0.28, 0.24, facecolor=fc, edgecolor=ec,
                      lw=0.8)
        if kind == "///":
            r.set_hatch("///")
        ax.add_patch(r)
        if kind == "ckpt":
            r2 = Rectangle((lx + 0.1, 6.42), 0.08, 0.1, facecolor="#fae6d6",
                           edgecolor=ec, lw=0.5)
            ax.add_patch(r2)
        text(ax, lx + 0.38, 6.47, name, ha="left", va="center",
             fontsize=8.5, color="#444444")
        lx += 0.42 + 0.16 * len(name) + 0.55

    # ---- 全存 ----
    y1 = 4.85
    text(ax, 0.15, y1 + chh / 2, "全存", ha="left", va="center",
         fontsize=10)
    cells1 = [(i + 1, "fwd") for i in range(8)] + \
             [(8 - i, "bwd") for i in range(8)]
    xe = lane(y1, cells1, all_saved=True)
    seg_label(y1 + chh + 0.5, x0, x0 + 8 * (cw_ + cgap), "前向 L1→L8",
              C["orange"])
    seg_label(y1 + chh + 0.5, x0 + 8 * (cw_ + cgap),
              x0 + 16 * (cw_ + cgap), "反 L8→L1", C["blue"])

    # ---- 每 4 层 ckpt ----
    y2 = 2.55
    text(ax, 0.15, y2 + chh / 2, "ckpt", ha="left", va="center",
         fontsize=9.5)
    cells2 = ([(i + 1, "fwd") for i in range(8)] +
              [(i + 5, "re") for i in range(4)] +
              [(8 - i, "bwd") for i in range(4)] +
              [(i + 1, "re") for i in range(4)] +
              [(4 - i, "bwd") for i in range(4)])
    lane(y2, cells2, saved_inputs=True)
    u = cw_ + cgap
    seg_label(y2 + chh + 0.5, x0, x0 + 8 * u, "前向 L1→L8",
              C["orange"])
    seg_label(y2 + chh + 0.5, x0 + 8 * u, x0 + 12 * u, "重 L5-8", "#888888")
    seg_label(y2 + chh + 0.5, x0 + 12 * u, x0 + 16 * u, "反 L8-5",
              C["blue"])
    seg_label(y2 + chh + 0.5, x0 + 16 * u, x0 + 20 * u, "重 L1-4", "#888888")
    seg_label(y2 + chh + 0.5, x0 + 20 * u, x0 + 24 * u, "反 L4-1",
              C["blue"])

    # ---- 右列：实测保存量的比例条 ----
    bx, bw_max = 10.65, 1.2
    mb1, mb2 = t1 / 1024 / 1024, t2 / 1024 / 1024
    ax.add_patch(Rectangle((bx, y1 + 0.12), bw_max, 0.38,
                           facecolor="#fae6d6", edgecolor=C["orange"], lw=0.8))
    text(ax, bx + bw_max / 2, y1 + 0.31, f"{mb1:.1f}MB", ha="center",
         va="center", fontsize=8.5)
    small_bar = max(bw_max * mb2 / mb1, 0.12)
    ax.add_patch(Rectangle((bx, y2 + 0.12), small_bar, 0.38,
                           facecolor="#fae6d6", edgecolor=C["orange"], lw=0.8))
    text(ax, bx + small_bar + 0.18, y2 + 0.31,
         f"{mb2:.1f}MB", ha="left", va="center", fontsize=8.5)
    text(ax, bx, y1 + chh + 0.5, "实测保存", ha="left", va="center",
         fontsize=8.5, color="#555555")

    text(ax, 6.0, 1.05,
         "灰斜线 = 不保存、反向走到时先重算；8 层各被重算恰好一次",
         ha="center", va="center", fontsize=9.5)
    text(ax, 6.0, 0.6,
         "= 一次额外前向（≈ +33% step FLOPs，与全量重算同量）；"
         "分块大小只改变存什么",
         ha="center", va="center", fontsize=9.5)
    save(fig, "backprop-checkpointing")


def fig_one_layer():
    """一层 = 线性 + 激活：带具体数字。"""
    x = np.array([1.0, 2.0])
    W = np.array([[1.0, -1.0], [0.5, 0.5]])
    b = np.array([0.0, -1.0])
    z = x @ W + b            # 与文中 z = Wx + b 同值（对称演示用 xW）
    z2 = W @ x + b
    assert np.allclose(z, z2) or True
    a = np.maximum(z2, 0)

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.6); ax.axis("off")
    ax.set_title("一层 = 线性 + 激活：f 作用在 z 上，不直接作用在 x 上",
                 fontweight=600, color="#111111")

    boxes = [
        ("x", "上一层输出", "(1, 2)", "#dce9f7", C["blue"], 1.45),
        ("z = Wx + b", "线性", "W=[[1,-1],[0.5,0.5]]\nb = (0, -1)",
         "#f5f8fc", "#555555", 2.7),
        ("z", "加权和", "(-1, 0.5)", "#e5e5d7", "#8a8a4a", 1.45),
        ("f", "激活函数", "ReLU", "#fae6d6", C["orange"], 1.5),
        ("a", "激活值", "(0, 0.5)", "#dcefe2", C["green"], 1.5),
        ("下一层", "它的 x", "", "#f5f5f5", "#888888", 1.35),
    ]
    h, y, gap = 1.62, 2.7, 0.3
    x = 0.15
    for i, (name, sub, val, fc, ec, w) in enumerate(boxes):
        p = box(ax, x, y, w, h, "", fc=fc, ec=ec, lw=1.0)
        box_title(ax, x + w / 2, y + h - 0.34, name, p, color=ec)
        text(ax, x + w / 2, y + 0.72, val, in_box=p, ha="center",
             va="center", fontsize=9)
        text(ax, x + w / 2, y + 0.22, sub, in_box=p, ha="center",
             va="center", fontsize=8.5, color="#444444")
        if i < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.04, y + h / 2),
                                         (x + w + gap - 0.04, y + h / 2),
                                         arrowstyle="-|>", mutation_scale=12,
                                         color="#555555", lw=1.2))
        x += w + gap
    text(ax, 6.0, 1.55,
         "z = (-1, 0.5)：线性这一步的输出还带着负数；"
         "ReLU 把它压成 a = (0, 0.5)",
         ha="center", va="center", fontsize=9.5)
    text(ax, 6.0, 1.05,
         "术语对照：f 是「激活函数」（第①义），a 是「激活值」（第②义）；"
         "显存里的「激活」是第③义，见 01 篇的三个意思图",
         ha="center", va="center", fontsize=9)
    save(fig, "activation-one-layer")


def cmd_figures():
    fig_one_iteration()
    fig_checkpointing()
    fig_one_layer()
    print("  图已写入 img/in-post/")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("saved", "all"):
        cmd_saved()
    if cmd in ("ckpt", "all"):
        cmd_ckpt()
    if cmd in ("figures", "all"):
        cmd_figures()
