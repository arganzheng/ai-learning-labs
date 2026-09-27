"""gen_gpt2_attention_heatmap.py — 文章 01 的图：GPT-2 small 真实 attention 权重，两个头的热力图。

选一句带指代的话，取第 5 层的两个头（一个看"上一个词"，一个把 "it" 指回名词），
输出 SVG 到博客 img/in-post/。运行一次即可，依赖 transformers（本地缓存的 gpt2）+ matplotlib。
"""
import sys, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

OUT = sys.argv[1] if len(sys.argv) > 1 else "/Users/argan/Code/arganzheng.github.com/img/in-post/transformer-01-gpt2-attention-heads.svg"
plt.rcParams["font.family"] = ["Heiti SC", "PingFang SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"

tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
text = "The cat sat on the mat because it was tired"
ids = tok(text, return_tensors="pt")
with torch.no_grad():
    out = model(**ids, output_attentions=True)
labels = [t.replace("Ġ", "") for t in tok.convert_ids_to_tokens(ids["input_ids"][0])]
T = len(labels)

# 自动挑两个头：(a) 最像"看上一个词"的头；(b) 在 "it" 这一行把最多权重给 "cat" 的头
best_prev, best_it = None, None
it_pos, cat_pos = labels.index("it"), labels.index("cat")
for layer, A in enumerate(out.attentions):
    A = A[0]  # [heads, T, T]
    for h in range(A.shape[0]):
        a = A[h]
        prev = sum(a[i, i - 1].item() for i in range(1, T)) / (T - 1)
        it2cat = a[it_pos, cat_pos].item()
        if best_prev is None or prev > best_prev[0]: best_prev = (prev, layer, h)
        if best_it is None or it2cat > best_it[0]: best_it = (it2cat, layer, h)
print("看上一个词的头:", best_prev, " it→cat 的头:", best_it)

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
for ax, (score, layer, h), title in zip(axes, [best_prev, best_it],
        ["一个头在看「上一个词」", "一个头把 it 指回 cat"]):
    a = out.attentions[layer][0, h].numpy()
    im = ax.imshow(a, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(T)); ax.set_yticks(range(T))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9); ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(f"{title}（第 {layer} 层第 {h} 头）", fontsize=12)
    ax.set_xlabel("被看的 token（key）", fontsize=10); ax.set_ylabel("正在看的 token（query）", fontsize=10)
    for i in range(T):
        for j in range(T):
            if a[i, j] >= 0.25: ax.text(j, i, f"{a[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if a[i, j] > 0.6 else "black")
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="注意力权重（每行和为 1）")
fig.suptitle(f"GPT-2 small 对「{text}」的两个真实 attention 头：下三角是 causal mask（看不到右边）", fontsize=12)
fig.savefig(OUT, bbox_inches="tight")
print("saved", OUT)
