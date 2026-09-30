"""LoRA 专题（03）：从训练到服务——adapter 文件、加载与合并的数值等价、量化底座的失配、多 adapter、参考模型、新 token。
https://arganzheng.life/lora-in-production-adapters-merging-multi-lora-and-serving.html

需要先运行 02_knobs.py r16_all r16_attn（本脚本读 out/ckpt/ 下的两个 adapter）。

    python 03_deploy.py                    # 全部：files merge quant multi ref tokens
    python 03_deploy.py files merge        # 按名字选
"""
import argparse
import copy
import json
import os
import time

import torch
from peft import LoraConfig, PeftModel, get_peft_model

import lolab

CK = os.path.join(lolab.OUT, "ckpt")
ALL = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def need(name):
    p = os.path.join(CK, name)
    if not os.path.isdir(p):
        raise SystemExit(f"缺 {p}：先运行 python 02_knobs.py {name}")
    return p


# ---------------- 实验 1：adapter 文件里有什么 ----------------
def exp_files(base):
    lolab.banner("实验 1：save_pretrained 存下来的 adapter 是什么")
    for name in ("r16_all", "r16_attn"):
        p = need(name)
        print(f"  {name}/")
        for f in sorted(os.listdir(p)):
            print(f"    {f:<32}{os.path.getsize(os.path.join(p, f)) / 2**20:>8.2f} MB")
        cfg = json.load(open(os.path.join(p, "adapter_config.json")))
        keys = ["base_model_name_or_path", "peft_type", "r", "lora_alpha", "target_modules", "use_rslora", "use_dora", "init_lora_weights", "peft_version"]
        print("    adapter_config.json 里决定加载行为的字段：", {k: cfg.get(k) for k in keys})
    from safetensors import safe_open
    with safe_open(os.path.join(CK, "r16_all", "adapter_model.safetensors"), "pt") as f:
        ks = list(f.keys())
        a = f.get_tensor(ks[0])
        print(f"  adapter_model.safetensors：{len(ks)} 个张量（24 层 × 7 个矩阵 × A、B），第一个键 {ks[0]}，形状 {tuple(a.shape)}，dtype {a.dtype}")
    total = sum(p.numel() for p in base.parameters())
    print(f"  对照：底座 {total / 1e6:.0f}M 参数，FP32 存 {total * 4 / 2**30:.2f} GiB、BF16 存 {total * 2 / 2**30:.2f} GiB；adapter 是它的几十分之一。"
          f" 8B 底座 BF16 16 GB，r=16 全部线性层的 adapter 42M × 2 B = 84 MB。\n")


# ---------------- 实验 2：加载 → 合并的数值等价 ----------------
def exp_merge(tok, base, evals):
    lolab.banner("实验 2：加载 adapter，与 merge_and_unload 前后的输出是否一样")
    m = PeftModel.from_pretrained(copy.deepcopy(base), need("r16_all"))
    m.eval()
    x = tok.apply_chat_template([{"role": "user", "content": "Give me three tips for better sleep."}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with torch.no_grad():
        l_lora = m(**x).logits
        t = time.time(); [m(**x) for _ in range(3)]; t_lora = (time.time() - t) / 3
        l_base = base(**x).logits
    v = lolab.completion_loss(m, tok, evals)
    print(f"  加载后验证回复 loss {v:.4f}（02 训完当场算的数应当一致）；与 base 的 logits 最大差 {(l_lora - l_base).abs().max().item():.3f}（adapter 确实生效了）")
    merged = m.merge_and_unload()
    n_lora = sum(1 for n, _ in merged.named_modules() if "lora" in n)
    with torch.no_grad():
        l_merged = merged(**x).logits
        t = time.time(); [merged(**x) for _ in range(3)]; t_merged = (time.time() - t) / 3
    print(f"  merge_and_unload 后：模型里 lora 模块 {n_lora} 个，参数量 {sum(p.numel() for p in merged.parameters()) / 1e6:.1f}M（与底座相同）")
    print(f"  合并前后 logits 最大差 {(l_merged - l_lora).abs().max().item():.2e}（浮点舍入量级）；一次前向 {t_lora * 1000:.0f} ms → {t_merged * 1000:.0f} ms")
    out = merged.generate(**x, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id)
    print("  生成：", repr(tok.decode(out[0][x["input_ids"].shape[1]:], skip_special_tokens=True))[:200])
    md = os.path.join(CK, "merged_r16_all")
    if not os.path.isdir(md):
        merged.save_pretrained(md); tok.save_pretrained(md)
    print(f"  合并后的模型存到 {os.path.relpath(md)}（{sum(os.path.getsize(os.path.join(md, f)) for f in os.listdir(md)) / 2**30:.2f} GiB，与底座一样大）\n")
    del m, merged


# ---------------- 实验 3：量化底座与 adapter 的失配 ----------------
def exp_quant(tok, base, evals):
    lolab.banner("实验 3：先合并再量化 vs 量化底座挂 adapter——NF4 上的失配有多大")
    try:
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float32)
        md = os.path.join(CK, "merged_r16_all")
        if not os.path.isdir(md):
            print("  需要先运行 merge 实验"); return
        _, base_nf4 = lolab.load(quantization_config=q, device_map=lolab.device())
        _, merged_nf4 = lolab.load(md, quantization_config=q, device_map=lolab.device())
    except Exception as e:
        print(f"  跳过（需要 bitsandbytes）：{type(e).__name__}: {str(e)[:150]}"); return
    print(f"  {'配置':<36}{'验证回复 loss':>14}")

    def row(name, m):                                # 当场算：PeftModel.from_pretrained 会就地改写底座，之后再算就不是"训练前"了
        print(f"  {name:<36}{lolab.completion_loss(m, tok, evals):>14.4f}")

    row("FP32 底座（训练前）", base)
    row("NF4 底座（训练前）", base_nf4)
    m_fp = PeftModel.from_pretrained(copy.deepcopy(base), need("r16_all"))
    row("FP32 底座 + adapter（训练时的组合）", m_fp)
    m_q = PeftModel.from_pretrained(base_nf4, need("r16_all"))
    row("NF4 底座 + 同一个 adapter", m_q)
    row("先合并再 NF4 量化", merged_nf4)
    print("  看什么：adapter 是对着某个底座学的；换成量化底座，差别 = 量化误差（训练前两行的差）而不是 adapter 失效；"
          "先合并再量化多一次量化误差落在合并后的权重上。QLoRA 训出的 adapter 反过来装到 BF16 底座上也是同一件事。\n")
    del m_fp, m_q, base_nf4, merged_nf4


# ---------------- 实验 4：多个 adapter ----------------
def exp_multi(tok, base, evals):
    lolab.banner("实验 4：一个底座挂两个 adapter：切换、加权合成、与 disable_adapter 当参考模型")
    m = PeftModel.from_pretrained(copy.deepcopy(base), need("r16_all"), adapter_name="all")
    m.load_adapter(need("r16_attn"), adapter_name="attn")
    print(f"  已加载 adapter：{list(m.peft_config)}，当前激活：{m.active_adapter}")
    base_v = lolab.completion_loss(base, tok, evals)
    for name in ("all", "attn"):
        m.set_adapter(name)
        print(f"  set_adapter({name!r:<7}) 验证回复 loss {lolab.completion_loss(m, tok, evals):.4f}")
    with m.disable_adapter():
        v = lolab.completion_loss(m, tok, evals)
    print(f"  with disable_adapter(): {v:.4f}（base 是 {base_v:.4f}）← DPO / GRPO 的参考模型就这么来，不用再放一份底座")
    for combo, kw in [("linear 0.5/0.5", dict(combination_type="linear", weights=[0.5, 0.5])),
                      ("cat（秩 16+16）", dict(combination_type="cat", weights=[1.0, 1.0])),
                      ("svd 到秩 16", dict(combination_type="svd", weights=[1.0, 1.0]))]:
        try:
            m.add_weighted_adapter(["all", "attn"], adapter_name=f"mix_{kw['combination_type']}", **kw)
            m.set_adapter(f"mix_{kw['combination_type']}")
            r = m.peft_config[f"mix_{kw['combination_type']}"].r
            print(f"  add_weighted_adapter {combo:<16} r={r:<3} 验证回复 loss {lolab.completion_loss(m, tok, evals):.4f}")
        except Exception as e:
            print(f"  add_weighted_adapter {combo}: 失败 {type(e).__name__}: {str(e)[:120]}")
    # 服务侧的账
    print(f"\n  服务侧的账（r=16 全部线性层）：一个 adapter 8.8M × 2 B = {8.8e6 * 2 / 2**20:.0f} MB，底座 BF16 {494e6 * 2 / 2**30:.2f} GiB——"
          f"挂 50 个 adapter 多 {50 * 8.8e6 * 2 / 2**30:.2f} GiB，存 50 份合并模型是 {50 * 494e6 * 2 / 2**30:.0f} GiB。"
          f" 8B：adapter 84 MB，底座 16 GB。vLLM 的 max_loras × max_lora_rank 预留的就是这块槽位。\n")
    del m


# ---------------- 实验 5：新加的 token 为什么学不会 ----------------
def exp_tokens(tok, base):
    lolab.banner("实验 5：加一个新 special token，只挂 LoRA 时它的 embedding 会不会动")
    tok2 = copy.deepcopy(tok)
    n_added = tok2.add_special_tokens({"additional_special_tokens": ["<|tool_call|>"]})
    new_id = tok2.convert_tokens_to_ids("<|tool_call|>")
    print(f"  新增 {n_added} 个 token，id {new_id}；模型 embedding 有 {base.get_input_embeddings().weight.shape[0]} 行（Qwen 预留了空位，不必 resize）")
    x = tok2("call <|tool_call|> now", return_tensors="pt").input_ids
    for name, cfg in [("LoRA r=16 全部线性层", LoraConfig(r=16, lora_alpha=32, target_modules=ALL, task_type="CAUSAL_LM")),
                      ("+ trainable_token_indices", LoraConfig(r=16, lora_alpha=32, target_modules=ALL, task_type="CAUSAL_LM", trainable_token_indices={"embed_tokens": [new_id]})),
                      ("+ modules_to_save=[embed_tokens, lm_head]", LoraConfig(r=16, lora_alpha=32, target_modules=ALL, task_type="CAUSAL_LM", modules_to_save=["embed_tokens", "lm_head"]))]:
        m = get_peft_model(copy.deepcopy(base), cfg)
        _, trainable = lolab.count_params(m)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
        out = m(input_ids=x, labels=x); out.loss.backward(); opt.step()
        with torch.no_grad():
            # 新 token 的输入 embedding 现在是什么：直接用模型的 embedding 层查一次
            row = m.get_input_embeddings()(torch.tensor([[new_id]]))[0, 0]
            base_row = base.get_input_embeddings().weight[new_id]
            moved = (row - base_row).abs().max().item()
        print(f"  {name:<42} 可训练 {trainable / 1e6:>6.1f}M   一步后新 token 的 embedding 最大改动 {moved:.2e}")
        del m, opt
    print("  看什么：LoRA 只挂在线性层上，embedding 与 lm_head 冻结，新 token 那一行永远是初始值——训练数据里的 <|tool_call|> 学不会。"
          "trainable_token_indices 只训那几行（几 KB），modules_to_save 训整张表（0.5B 上是 136M，比 LoRA 本身大 15 倍）。\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="*", choices=["files", "merge", "quant", "multi", "tokens"])
    args = ap.parse_args()
    exps = args.experiments or ["files", "merge", "quant", "multi", "tokens"]
    lolab.seed(0)
    print(f"设备 {lolab.device()}")
    tok, base = lolab.load()
    base.eval()
    _, evals = lolab.load_data(64, 100)
    if "files" in exps:
        exp_files(base)
    if "merge" in exps:
        exp_merge(tok, base, evals)
    if "quant" in exps:
        exp_quant(tok, base, evals)
    if "multi" in exps:
        exp_multi(tok, base, evals)
    if "tokens" in exps:
        exp_tokens(tok, base)


if __name__ == "__main__":
    main()
