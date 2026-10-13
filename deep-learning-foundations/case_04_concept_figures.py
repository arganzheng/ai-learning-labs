"""案例（DL 04）：正则化与泛化——文章概念图与工作实例，把第四章的数轴、示意曲线、
最小范数解、label smoothing 与小语言模型记忆曲线画成图，并打印文中引用的数字。
https://arganzheng.life/regularization-and-generalization.html

    python case_04_concept_figures.py   # 纯概念图，无训练，秒级

输出 out/reg-*.svg。
"""
import numpy as np

from _plot import C, plt, save

BLUE, ORANGE, GREEN, RED, GRAY, PURPLE = (
    C["blue"], C["orange"], C["green"], C["red"], C["gray"], C["purple"])
LW = 2.2


def nd_axis():
    """第一章：N/D 对数刻度数轴与两个区间。"""
    pts = [("Llama-3 预训练", 0.0005), ("Chinchilla 最优", 0.05),
           ("参数 = 数据", 1.0), ("奖励模型", 160),
           ("本文实验", 407), ("SFT", 1600)]
    print("=== N/D 数轴 ===")
    for name, v in pts:
        print(f"  {name:<14} N/D = {v}")

    fig, ax = plt.subplots(figsize=(10.6, 3.1))
    ax.set_xscale("log")
    ax.set_xlim(2e-4, 8e3)
    ax.set_ylim(0.05, 1.05)
    ax.axhline(0.42, color="#222222", lw=1.4)
    ax.set_yticks([])
    ax.set_xlabel("参数 / 数据比 $N / D$（对数刻度）", fontsize=10)
    ax.spines[["left", "right", "top"]].set_visible(False)

    ax.axvspan(2e-4, 1, ymin=0.30, ymax=0.54, color=BLUE, alpha=0.14)
    ax.axvspan(1, 8e3, ymin=0.30, ymax=0.54, color=ORANGE, alpha=0.14)
    ax.text(1.4e-2, 0.86, "数据主导区间：背不下数据，\n过拟合不是主要风险",
            ha="center", fontsize=10, color=BLUE, weight="bold")
    ax.text(280, 0.86, "过参数化区间：能背下数据，\n会不会过拟合看训多久、有没有噪声",
            ha="center", fontsize=10, color=ORANGE, weight="bold")

    # 上下交错标注，避免重叠；只给上面的标注留一层高度
    layout = {0: -1, 1: -1, 2: 1, 3: -1, 4: 1, 5: -1}
    for i, (name, v) in enumerate(pts):
        side = layout[i]
        ax.plot([v], [0.42], "o", ms=6,
                color=BLUE if v <= 1 else ORANGE)
        ax.annotate(f"{name}\n$N/D$ = {v:g}", (v, 0.42),
                    xytext=(v, 0.52 if side > 0 else 0.30),
                    ha="center", va="bottom" if side > 0 else "top",
                    fontsize=9, color="#333333",
                    arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8))
    save(fig, "reg-nd-axis")


def double_descent_schematic():
    """第三章：测试误差三段示意曲线。"""
    cap = np.linspace(0, 10, 500)
    # U 形 + 插值阈值尖峰 + 第二次下降（示意）
    test = (1.2 * np.exp(-0.9 * cap)
            + 2.6 * np.exp(-0.5 * (cap - 3.6) ** 2 / 0.28)
            + 0.55 / (1 + np.exp(1.4 * (cap - 4.2)))
            + 0.18)
    train = 0.85 / (1 + np.exp(2.2 * (cap - 3.6))) + 0.02
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.plot(cap, test, color=BLUE, lw=LW, label="测试误差")
    ax.plot(cap, train, color=ORANGE, lw=LW, label="训练误差")
    ax.axvline(3.6, color=GRAY, lw=0.9)
    ax.annotate("插值阈值：\n训练误差刚到 0", (3.6, 2.85),
                xytext=(4.6, 3.05), ha="left", fontsize=9.5, color=GRAY,
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
    ax.text(1.15, 2.35, "欠拟合：经典 U 形", ha="center", fontsize=10.5,
            color="#333333")
    ax.text(6.6, 1.35, "过参数化：\n第二次下降", ha="center", fontsize=10.5,
            color="#333333")
    ax.set_xlabel("模型容量（参数 / 样本数）")
    ax.set_ylabel("误差")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylim(-0.05, 3.4)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_title("double descent：测试误差的三段", fontsize=12)
    save(fig, "reg-double-descent-schematic")


def min_norm():
    """第四章工作实例 a：最小范数解。"""
    x = np.array([1.0, 2.0]); y = 5.0
    w = np.zeros(2); lr = 0.05; steps = 50
    traj = [w.copy()]
    for _ in range(steps):
        g = 2 * (w @ x - y) * x
        w = w - lr * g
        traj.append(w.copy())
    traj = np.array(traj)
    w_min = x * (y / (x @ x))           # = (1, 2)
    w_other = np.array([5.0, 0.0])
    print("=== 最小范数解 ===")
    print(f"  解集：w1 + 2 w2 = 5；GD 收敛到 w* = {w.round(3)}，"
          f"理论最小范数点 {w_min}，‖w*‖ = {np.linalg.norm(w_min):.3f}（√5）")
    print(f"  另一个插值解 {w_other}，范数 {np.linalg.norm(w_other):.3f}")

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    t = np.linspace(-1.5, 6.5, 100)
    ax.plot(t, (5 - t) / 2, color=GRAY, lw=1.6)
    ax.text(6.2, -0.95, "$w_1 + 2 w_2 = 5$（所有插值解）",
            fontsize=9.5, color=GRAY, ha="center")
    ax.plot(traj[:, 0], traj[:, 1], color=BLUE, lw=LW)
    ax.plot(traj[::6, 0], traj[::6, 1], "o", ms=4, color=BLUE)
    ax.plot(0, 0, "o", ms=7, color="#222222")
    ax.text(-0.05, -0.28, "初始化 $(0,0)$", fontsize=9)
    ax.plot(*w_min, "o", ms=8, color=ORANGE)
    ax.text(w_min[0] + 0.12, w_min[1] + 0.18,
            f"最小范数解 $w^*=(1,2)$\n$\\|w^*\\|=\\sqrt{{5}}\\approx2.236$",
            fontsize=9.5, color=ORANGE)
    ax.plot(*w_other, "o", ms=8, color=RED)
    ax.text(w_other[0] + 0.15, w_other[1] + 0.42,
            "另一个插值解 $(5,0)$，范数 $=5$", fontsize=9.5, color=RED,
            ha="left")
    ax.set_xlabel("$w_1$"); ax.set_ylabel("$w_2$")
    ax.set_xlim(-1.2, 6.8); ax.set_ylim(-1.2, 4.6)
    ax.set_aspect("equal")
    ax.set_title("GD 从小初始化出发，收敛到最小范数解", fontsize=11)
    save(fig, "reg-min-norm")


def label_smoothing():
    """第五章工作实例 b：label smoothing 的 loss–gap 曲线。"""
    Cls, eps = 10, 0.1
    g = np.linspace(0, 10, 400)
    logits = np.zeros((len(g), Cls)); logits[:, 0] = g
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    ce_onehot = -np.log(p[:, 0])
    target = np.full(Cls, eps / (Cls - 1)); target[0] = 1 - eps
    ce_smooth = -(np.log(p) * target).sum(1)
    g_star = np.log((1 - eps) * (Cls - 1) / eps)
    print("=== label smoothing ===")
    print(f"  C={Cls}，ε={eps}：正确类目标 {1 - eps}，其余每类 "
          f"{eps / (Cls - 1):.4f}")
    print(f"  平滑后 CE 的最小点在 g* = ln({(1 - eps) * (Cls - 1) / eps:.0f}) "
          f"= {g_star:.3f}；one-hot 随 g→∞ 趋 0")

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.plot(g, ce_onehot, color=BLUE, lw=LW, label="one-hot 目标")
    ax.plot(g, ce_smooth, color=ORANGE, lw=LW,
            label="平滑目标（ε = 0.1）")
    i = np.argmin(np.abs(g - g_star))
    ax.plot(g[i], ce_smooth[i], "o", ms=7, color=ORANGE)
    ax.annotate(f"最小点 $g^*=\\ln 81\\approx{g_star:.3f}$",
                (g[i], ce_smooth[i]), xytext=(g[i] + 1.6, ce_smooth[i] + 0.9),
                fontsize=10, color=ORANGE,
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.9))
    ax.set_xlabel("正确类与错误类的 logit 差 $g$")
    ax.set_ylabel("交叉熵 loss")
    ax.legend(fontsize=10)
    ax.set_title("one-hot 下 loss 逼着 logit 差走向无穷；平滑后有最小点",
                 fontsize=11)
    save(fig, "reg-label-smoothing")


def lm_20k():
    """第七章：小语言模型 2 万字符组的逐 epoch 曲线（数字来自第九章的表）。"""
    ep = np.array([1, 4, 8, 16, 24, 32, 40])
    tr = np.array([3.166, 2.353, 1.790, 1.213, 0.850, 0.606, 0.438])
    ho = np.array([3.091, 2.596, 2.431, 2.408, 2.527, 2.734, 2.975])
    mem = np.array([0.0, 0.5, 2.5, 5.5, 5.5, 7.0, 10.0])
    best = int(np.argmin(ho))
    print("=== 小语言模型 2 万字符 ===")
    for i in range(len(ep)):
        print(f"  ep {ep[i]:>2}：训练 {tr[i]:.3f}  held-out {ho[i]:.3f}  "
              f"逐字记忆率 {mem[i]:.1f}%")
    print(f"  held-out 最低点在第 {ep[best]} epoch（{ho[best]:.3f}），"
          f"此后升到 {ho[-1]:.3f}；训练 loss 继续降到 {tr[-1]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.2))
    axes[0].plot(ep, tr, color=BLUE, lw=LW, label="训练 loss")
    axes[0].plot(ep, ho, color=ORANGE, lw=LW, label="held-out loss")
    axes[0].plot(ep[best], ho[best], "o", ms=8, color=ORANGE)
    axes[0].annotate(f"最低点：第 {ep[best]} epoch",
                     (ep[best], ho[best]), xytext=(ep[best] - 9, ho[best] - 0.55),
                     fontsize=10, color=ORANGE,
                     arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.9))
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("2 万字符：held-out 先降后升，训练 loss 继续降",
                      fontsize=11)
    axes[0].legend(fontsize=10)
    axes[1].plot(ep, mem, color=GREEN, lw=LW)
    axes[1].plot(ep, mem, "o", ms=4, color=GREEN)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("逐字记忆率 %")
    axes[1].set_title("同一时刻记忆率一路上升到 10%", fontsize=11)
    save(fig, "reg-lm-20k")


if __name__ == "__main__":
    nd_axis()
    double_descent_schematic()
    min_norm()
    label_smoothing()
    lm_20k()
