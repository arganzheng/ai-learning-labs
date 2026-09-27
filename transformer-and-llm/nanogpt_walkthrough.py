"""nanogpt_walkthrough.py — 文章 03《手搓 GPT（上）：nanoGPT model.py 逐行解析》的配套实验。

  1. 用一个极小配置实例化 GPT，逐 module 打印参数形状与参数量（对照文章第六章的表）
  2. 用 forward hook 打印一次前向里每一站的形状
  3. GPT-2 small：get_num_params() = 123.65M（不含 wpe）；加回 wpe = 124.44M
  4. from_pretrained('gpt2') 与 HuggingFace GPT2LMHeadModel 的 logits 对拍
  5. generate()：温度 / top-k 的效果；同一 prompt 贪心 vs 采样

运行：python nanogpt_walkthrough.py [--no-hf]    依赖：torch；第 4 步需要 transformers 与本地缓存的 gpt2
"""
import os, sys, torch, torch.nn.functional as F
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1"); os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
from nanogpt_model import GPT, GPTConfig

torch.manual_seed(0)
def section(t): print(f"\n== {t}")

# 1. 极小配置：结构与参数量
section("1. 极小配置 GPTConfig(vocab 16, block 8, 2 层 2 头 d=8) 的全部参数")
cfg = GPTConfig(vocab_size=16, block_size=8, n_layer=2, n_head=2, n_embd=8, dropout=0.0, bias=True)
m = GPT(cfg)
for name, p in m.named_parameters():
    print(f"  {name:36s} {str(tuple(p.shape)):12s} {p.numel():5d}")
print(f"  合计 {sum(p.numel() for p in m.parameters()):,}（lm_head.weight 与 wte.weight 是同一个张量，只数一次）")

# 2. 一次前向的形状
section("2. 一次前向 idx [2, 5] 经过每一站的形状")
shapes = []
def hook(name):
    def f(mod, inp, out): shapes.append((name, tuple(out.shape)))
    return f
hs = [m.transformer.wte.register_forward_hook(hook("transformer.wte")), m.transformer.wpe.register_forward_hook(hook("transformer.wpe"))]
for i, blk in enumerate(m.transformer.h):
    hs += [blk.attn.c_attn.register_forward_hook(hook(f"h.{i}.attn.c_attn")), blk.attn.register_forward_hook(hook(f"h.{i}.attn")),
           blk.mlp.c_fc.register_forward_hook(hook(f"h.{i}.mlp.c_fc")), blk.mlp.register_forward_hook(hook(f"h.{i}.mlp")), blk.register_forward_hook(hook(f"h.{i}"))]
hs += [m.transformer.ln_f.register_forward_hook(hook("transformer.ln_f")), m.lm_head.register_forward_hook(hook("lm_head"))]
idx = torch.randint(0, 16, (2, 6)); logits, loss = m(idx[:, :-1].contiguous(), idx[:, 1:].contiguous())   # 切片不连续，nanoGPT 的 targets.view(-1) 会报错；train.py 里 y 是 stack 出来的所以连续
for h in hs: h.remove()
for n, s in shapes: print(f"  {n:24s} {s}")
print(f"  loss = {loss.item():.4f}（≈ ln 16 = 2.7726）")
logits_inf, _ = m(idx[:, :-1])
print(f"  不传 targets 时 logits 形状 {tuple(logits_inf.shape)}：只算最后一个位置（forward 里的 x[:, [-1], :]）")

# 3. GPT-2 small 的参数量
section("3. GPT-2 small 的参数量")
g = GPT(GPTConfig(vocab_size=50257, block_size=1024, n_layer=12, n_head=12, n_embd=768, bias=True))
print(f"  get_num_params()            = {g.get_num_params():,}   （默认扣掉 wpe 的 786,432）")
print(f"  get_num_params(False)       = {g.get_num_params(False):,}   （文章 01 表里的 124,439,808）")

# 4. 与 HuggingFace 对拍
if "--no-hf" not in sys.argv:
    section("4. from_pretrained('gpt2') 与 HuggingFace 的 logits 对拍")
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    ids = tok("The cat sat on the mat because it was", return_tensors="pt")["input_ids"]
    nano = GPT.from_pretrained("gpt2").eval()
    hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    with torch.no_grad():
        l_nano, _ = nano(ids); l_hf = hf(ids).logits[:, -1:, :]
    print(f"  最后一个位置 logits：最大绝对差 {(l_nano - l_hf).abs().max().item():.2e}，logits 量级 {l_hf.abs().max().item():.0f}，"
          f"相对差 {((l_nano - l_hf).abs().max() / l_hf.abs().max()).item():.1e}；argmax 相同: {bool((l_nano.argmax(-1) == l_hf.argmax(-1)).all())}")
    top = torch.topk(F.softmax(l_nano[0, -1], -1), 5)
    print("  下一个 token 的前五名:", [(tok.decode([i]), round(p.item(), 3)) for p, i in zip(top.values, top.indices)])
    # 5. generate
    section("5. generate：同一 prompt，贪心 vs 采样（temperature / top_k）")
    prompt = tok("The meaning of life is", return_tensors="pt")["input_ids"]
    torch.manual_seed(1)
    for name, kw in [("温度 1e-4（≈贪心）", dict(temperature=1e-4)), ("温度 1.0 + top_k 50", dict(temperature=1.0, top_k=50)), ("温度 1.5 + top_k 50", dict(temperature=1.5, top_k=50))]:
        out = nano.generate(prompt, max_new_tokens=12, **kw)
        print(f"  {name:22s} → {tok.decode(out[0])!r}")
