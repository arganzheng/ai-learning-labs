"""多模态（05）语音下篇：半双工 vs 全双工的时间线示意图（参数取自文中的公开数字）。"""
import numpy as np

from _plot import C, plt, save

fig, axes = plt.subplots(2, 1, figsize=(7.6, 3.6), gridspec_kw={"height_ratios": [1, 1.3]})

# ---- 半双工：串联管线
ax = axes[0]
ax.barh(1, 2.0, left=0, color=C["blue"], height=0.5); ax.text(1.0, 1, "用户说话 $2.0\,\mathrm{s}$", ha="center", va="center", color="white", fontsize=10.5)
segs = [("VAD 0.5", 0.5), ("ASR 0.3", 0.3), ("LLM 0.6", 0.6), ("TTS 0.4", 0.4)]
t = 2.0
label_y = [0.55, 1.0, 0.55, 1.0]                # 错两行放条上方空隙，上行用引线指到条
for i, (name, w) in enumerate(segs):
    ax.barh(0, w, left=t, color=C["light"], height=0.5, edgecolor="white")
    ax.text(t + w / 2, label_y[i], name, ha="center", va="center", fontsize=10.5)
    if label_y[i] > 0.6:
        ax.plot([t + w / 2] * 2, [0.27, 0.8], color="#aaaaaa", lw=0.8)
    t += w
ax.barh(0, 1.6, left=t, color=C["orange"], height=0.5); ax.text(t + 0.8, 0, "模型说话", ha="center", va="center", color="white", fontsize=10.5)
ax.annotate("", xy=(t, -0.45), xytext=(2.0, -0.45), arrowprops=dict(arrowstyle="<->", color=C["red"], lw=1))
ax.text((2.0 + t) / 2, -0.58, f"时延 ${t - 2.0:.1f}\,\mathrm{{s}}$", ha="center", va="top", color=C["red"], fontsize=10.5)
ax.set_yticks([0, 1]); ax.set_yticklabels(["模型", "用户"]); ax.set_xlim(0, 6); ax.set_ylim(-1.1, 1.75); ax.set_title("半双工：等静音（VAD）$\\to$ ASR $\\to$ LLM $\\to$ TTS 串联，用户说完才开始算（单位：秒）", fontsize=11.5)
ax.set_xlabel("秒")

# ---- 全双工：两条流每 80 ms 一格
ax = axes[1]
frame = 0.08; n = int(6 / frame)
user = np.zeros(n, bool); user[:25] = True; user[45:52] = True                         # 用户 0–2.0 s 说话；3.6–4.16 s 插话
model = np.zeros(n, bool); model[27:45] = True; model[54:] = True                        # 用户停 2 帧后开始说；被插话后停，再等 2 帧继续
for i in range(n):
    ax.add_patch(plt.Rectangle((i * frame, 0.75), frame, 0.5, color=C["blue"] if user[i] else "#e9eef5", ec="white", lw=0.3))
    ax.add_patch(plt.Rectangle((i * frame, -0.25), frame, 0.5, color=C["orange"] if model[i] else "#f6ecdc", ec="white", lw=0.3))
ax.text(1.0, 1.0, "用户说话", ha="center", va="center", color="white", fontsize=10.5)
ax.text(3.6 + 0.28, 1.0, "插话", ha="center", va="center", color="white", fontsize=10.5)
ax.text(2.9, 0, "模型说话", ha="center", va="center", color="white", fontsize=10.5)
ax.text(5.1, 0, "继续说", ha="center", va="center", color="white", fontsize=10.5)
ax.text(1.95, -0.62, "静音 token（每 $80\,\mathrm{ms}$ 一个，一直在输出）", ha="right", fontsize=10.5, color=C["gray"])
ax.annotate("", xy=(27 * frame, -0.4), xytext=(25 * frame, -0.4), arrowprops=dict(arrowstyle="<->", color=C["red"], lw=1))
ax.text(26 * frame + 0.05, -0.9, "2 帧 $=160\,\mathrm{ms}$", color=C["red"], fontsize=10.5, ha="left")
ax.annotate("", xy=(46 * frame, 0.5), xytext=(46 * frame, 0.25), arrowprops=dict(arrowstyle="->", color=C["red"], lw=1))
ax.text(46 * frame + 0.05, 0.42, "用户一开口，下一帧就静音", color=C["red"], fontsize=10.5)
ax.set_yticks([0, 1]); ax.set_yticklabels(["模型流", "用户流"]); ax.set_xlim(0, 6); ax.set_ylim(-1.1, 1.4)
ax.set_title("全双工（Moshi）：两条流同步前进，每一帧都在决定「说还是不说」", fontsize=11.5); ax.set_xlabel("秒")
for a in axes:
    a.spines["left"].set_visible(False)
save(fig, "05-duplex-timeline")
