"""《向量、矩阵与形状》第六章：W_Q / W_K 为什么要先投影、W_O 在多头里做什么。

全部是手算尺寸的玩具数字，文章里的表和图都由这里输出：

    python 01_attention_projections.py            # 打印数字（与 expected/01_attention_projections.txt 对照）
    python 01_attention_projections.py out.svg    # 另外生成 W_O 配图
"""
import sys

import numpy as np

# ---------- 1. 打分前为什么要投影：x_i W_Q (x_j W_K)^T = x_i M x_j^T ----------
# 二维玩具 embedding：第 0 维 = "名词性"，第 1 维 = "代词性"
TOK = {"猫": np.array([1.0, 0.0]), "它": np.array([0.0, 1.0])}
W_Q = np.array([[0.0, 0.0],   # 名词性 → 不找任何东西
                [1.0, 0.0]])  # 代词性 → 去找名词性
W_K = np.eye(2)

# ---------- 2. W_O：两个头的输出拼接后，W_O 的两块各自写回整条残差流 ----------
X = np.array([1.0, 0.0, 2.0, 1.0])          # 第六章六步表里的那个 x（d = 4）
O1 = np.array([1.0, 2.0])                   # 头 1 的 attention 输出（d_h = 2）
O2 = np.array([3.0, -1.0])                  # 头 2 的 attention 输出
W_O = np.array([[1.0, 0.0, 1.0, 0.0],       # 前两行 = W_O¹，只乘头 1 的两个数
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],       # 后两行 = W_O²，只乘头 2 的两个数
                [1.0, 0.0, 0.0, 0.0]])


def scores(wq, wk):
    names = list(TOK)
    return names, np.array([[(TOK[a] @ wq) @ (TOK[b] @ wk) for b in names] for a in names])


def fmt(v):
    return "[" + ", ".join(f"{x:g}" for x in v) + "]"


def report():
    print("== 1. 打分：不投影 vs 投影 ==")
    for label, wq, wk in [("不投影（W_Q = W_K = I）", np.eye(2), np.eye(2)), ("投影后", W_Q, W_K)]:
        names, s = scores(wq, wk)
        print(f"  {label}：行 = 谁在找（query），列 = 被找的（key）")
        for a, row in zip(names, s):
            print("    " + a + " → " + "  ".join(f"{b}:{x:g}" for b, x in zip(names, row)))
    print("  M = W_Q W_K^T =", (W_Q @ W_K.T).tolist(),
          "是否对称:", bool(np.allclose(W_Q @ W_K.T, (W_Q @ W_K.T).T)))

    print("\n== 2. W_O ==")
    cat = np.concatenate([O1, O2])
    p1, p2 = O1 @ W_O[:2], O2 @ W_O[2:]
    out = cat @ W_O
    assert np.allclose(out, p1 + p2)
    print("  拼接 o =", fmt(cat))
    print("  o¹ W_O¹ =", fmt(p1), " o² W_O² =", fmt(p2))
    print("  o W_O =", fmt(out), "= 两项之和:", bool(np.allclose(out, p1 + p2)))
    print("  不经 W_O 直接加回: x + o =", fmt(X + cat))
    print("  经 W_O 加回:     h = x + o W_O =", fmt(X + out))

    d, h, dh = 4096, 32, 128
    fact = h * (d * dh + dh * d)
    merged = h * d * d
    print("\n== 3. 把每个头的 W_V^i W_O^i 合成一个 d×d 矩阵（Llama-3-8B 一层，按 32 个 query 头算）==")
    print(f"  分开存：32 × (4096×128 + 128×4096) = {fact / 1e6:.1f} M")
    print(f"  合成存：32 × 4096×4096           = {merged / 1e6:.1f} M（{merged // fact} 倍）")
    return cat, p1, p2, out


# ---------- SVG ----------
BLUE = ("#eef4fb", "#1f6fb2")
RED = ("#fdeeee", "#c0392b")
GRAY = ("#f4f4f4", "#888888")
PURP = ("#f3eefa", "#7d4fb2")
CW, CH = 36, 26


def svg(path, cat, p1, p2, out):
    el = []

    def text(x, y, s, size=11, anchor="middle", weight="normal", fill="#333"):
        el.append(f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{s}</text>')

    def cells(x, y, vals, colors):
        for i, (v, c) in enumerate(zip(vals, colors)):
            el.append(f'<rect x="{x + i * CW}" y="{y}" width="{CW}" height="{CH}" fill="{c[0]}" stroke="{c[1]}" stroke-width="1"/>')
            text(x + i * CW + CW / 2, y + 17.5, f"{v:g}".replace("-", "−"), 12)

    def arrow(x1, y1, x2, y2, label=None):
        el.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#555" stroke-width="1.5" marker-end="url(#ar)"/>')
        if label:
            text((x1 + x2) / 2, min(y1, y2) - 7, label, 10)

    text(380, 22, "W_O：两个头的输出先拼接，再由 W_O 的两块各自写回整条残差流、相加", 14, weight="600", fill="#222")
    # 左：两个头的输出
    text(30, 62, "头 1 输出 o¹", 12, "start", "600")
    cells(30, 70, O1, [BLUE] * 2)
    text(30, 132, "头 2 输出 o²", 12, "start", "600")
    cells(30, 140, O2, [RED] * 2)
    arrow(108, 83, 160, 108)
    arrow(108, 153, 160, 128)
    # 拼接
    text(166, 100, "拼接 o　[1, 4]", 12, "start", "600")
    cells(166, 108, cat, [BLUE, BLUE, RED, RED])
    text(238, 152, "头 1 只占前两格、头 2 只占后两格", 10, fill="#666")
    arrow(316, 121, 352, 121, "× W_O")
    # W_O
    text(434, 48, "W_O　[4, 4]", 12, weight="600")
    for r in range(4):
        cells(362, 56 + r * CH, W_O[r], [BLUE if r < 2 else RED] * 4)
    text(510, 82, "W_O¹", 11, "start", "600", BLUE[1])
    text(510, 134, "W_O²", 11, "start", "600", RED[1])
    text(434, 182, "o W_O = o¹ W_O¹ + o² W_O²", 11, fill="#444")
    # 右：两项、求和、加残差
    arrow(548, 121, 584, 121)
    rx = 594
    text(rx, 48, "头 1 写回：o¹ W_O¹", 11, "start", "600", BLUE[1])
    cells(rx, 56, p1, [BLUE] * 4)
    text(rx, 100, "头 2 写回：o² W_O²", 11, "start", "600", RED[1])
    cells(rx, 108, p2, [RED] * 4)
    el.append(f'<line x1="{rx}" y1="142" x2="{rx + 4 * CW}" y2="142" stroke="#555" stroke-width="1"/>')
    text(rx - 8, 163, "相加", 11, "end")
    cells(rx, 146, out, [PURP] * 4)
    text(rx - 8, 200, "+ x", 12, "end")
    cells(rx, 184, X, [GRAY] * 4)
    text(rx - 8, 238, "= h", 12, "end", "600")
    cells(rx, 222, X + out, [PURP] * 4)
    # 底部说明
    notes = [
        "没有 W_O，直接把拼接结果加回 x：头 1 只能改残差流的第 0、1 维，头 2 只能改第 2、3 维——",
        "可残差流的维度并不按头分工，下一层也不知道「第几格来自哪个头」。",
        "有了 W_O：每个头都能写到全部 4 维；第 2 维同时收到两个头的 1 和 3，第 0 维里两个头的 +1 与 −1 抵消。",
    ]
    for i, s in enumerate(notes):
        text(30, 286 + i * 20, s, 11, "start", fill="#444")
    head = ('<svg xmlns="http://www.w3.org/2000/svg" width="760" height="350" viewBox="0 0 760 350" '
            "font-family=\"-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', "
            "'Microsoft YaHei', 'Noto Sans CJK SC', sans-serif\" font-size=\"13\">\n"
            '<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
            'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs>\n'
            '<rect width="760" height="350" fill="#fff"/>\n')
    with open(path, "w", encoding="utf-8") as f:
        f.write(head + "\n".join(el) + "\n</svg>\n")


if __name__ == "__main__":
    res = report()
    if len(sys.argv) > 1:
        svg(sys.argv[1], *res)
