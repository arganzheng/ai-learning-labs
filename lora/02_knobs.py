"""LoRA 专题（02）：每个旋钮各让模型变成什么——r、target_modules、alpha / rsLoRA、lr、DoRA、PiSSA、LoRA+、QLoRA 与全量的对照。
https://arganzheng.life/lora-hyperparameters-rank-targets-alpha-lr-and-variants.html

模型 Qwen/Qwen2.5-0.5B（base），数据 HuggingFaceH4/no_robots 的 800 条单轮样本，每种配置各训 80 步（batch 4 × 512），
然后在 100 条验证样本的回复上算 loss、在 60 段 wikitext 上算 loss（遗忘）。CPU（8 线程）上每种配置约 12 分钟，全部约 2.5 小时。

    python 02_knobs.py                     # 全部配置
    python 02_knobs.py r16_all full        # 按名字选
    python 02_knobs.py --quick             # 每种 8 步、64 条数据，只确认环境
    python 02_knobs.py --list              # 列出配置名
结果追加写入 out/knobs.json，03_deploy.py 与画图脚本读它。
"""
import argparse
import copy
import os

import torch
from peft import LoraConfig, get_peft_model

import lolab

ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
ALL = ATTN + ["gate_proj", "up_proj", "down_proj"]


def lcfg(r, targets, alpha=None, **kw):
    return LoraConfig(r=r, lora_alpha=alpha if alpha is not None else 2 * r, target_modules=targets,
                      lora_dropout=0.0, task_type="CAUSAL_LM", **kw)


# 名字 → (说明, LoraConfig 或 None=全量, lr, 额外选项)
CONFIGS = {
    "full":        ("全量 lr 1e-5",                         None,                                      1e-5, {}),
    "r16_attn":    ("LoRA r=16 attention lr 1e-4",         lcfg(16, ATTN),                            1e-4, {"save": True}),
    "r16_all":     ("LoRA r=16 全部线性层 lr 1e-4",          lcfg(16, ALL),                             1e-4, {"save": True}),
    "r4_all":      ("LoRA r=4 全部线性层 lr 1e-4",           lcfg(4, ALL),                              1e-4, {}),
    "r64_all":     ("LoRA r=64 alpha=128（缩放 2）lr 1e-4",  lcfg(64, ALL),                             1e-4, {}),
    "r64_a32":     ("LoRA r=64 alpha=32（缩放 0.5）lr 1e-4",  lcfg(64, ALL, alpha=32),                   1e-4, {}),
    "r64_rslora":  ("rsLoRA r=64 alpha=32（缩放 4）lr 1e-4",  lcfg(64, ALL, alpha=32, use_rslora=True),  1e-4, {}),
    "r16_lr1e-5":  ("LoRA r=16 全部线性层 lr 1e-5",          lcfg(16, ALL),                             1e-5, {}),
    "r16_lr1e-3":  ("LoRA r=16 全部线性层 lr 1e-3",          lcfg(16, ALL),                             1e-3, {}),
    "r16_dora":    ("DoRA r=16 全部线性层 lr 1e-4",          lcfg(16, ALL, use_dora=True),              1e-4, {}),
    "r16_pissa":   ("PiSSA r=16 全部线性层 lr 1e-4",         lcfg(16, ALL, init_lora_weights="pissa_niter_4"), 1e-4, {}),
    "r16_loraplus": ("LoRA+ r=16 全部线性层 lr 1e-4（B ×4）", lcfg(16, ALL),                             1e-4, {"loraplus": 4}),
    "r16_qlora":   ("QLoRA NF4 r=16 全部线性层 lr 1e-4",     lcfg(16, ALL),                             1e-4, {"nf4": True}),
}


def build(tok, base, cfg, opts):
    """按配置得到可训练的模型（与可选的自定义优化器）。"""
    if opts.get("nf4"):
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float32)
        _, m = lolab.load(quantization_config=q, device_map=lolab.device())
        m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=False)
        return get_peft_model(m, cfg), None
    if cfg is None:
        return copy.deepcopy(base), None
    m = get_peft_model(copy.deepcopy(base), cfg)
    opt = None
    if opts.get("loraplus"):
        from peft.optimizers import create_loraplus_optimizer
        opt = create_loraplus_optimizer(model=m, optimizer_cls=torch.optim.AdamW, lr=1e-4,
                                        loraplus_lr_ratio=opts["loraplus"], weight_decay=0.0)
    return m, opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", choices=list(CONFIGS))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--steps", type=int, default=80)
    args = ap.parse_args()
    if args.list:
        for k, (d, *_) in CONFIGS.items():
            print(f"{k:<14}{d}")
        return
    names = args.names or list(CONFIGS)
    steps = 8 if args.quick else args.steps
    lolab.seed(0)
    print(f"设备 {lolab.device()}，torch 线程 {torch.get_num_threads()}，每种配置 {steps} 步")
    tok, base = lolab.load()
    total, _ = lolab.count_params(base)
    train, evals = lolab.load_data(64, 10) if args.quick else lolab.load_data()
    texts = lolab.load_plain_text(6 if args.quick else 60)
    results = {}
    if not args.quick and os.path.exists(os.path.join(lolab.OUT, "knobs.json")):
        results = lolab.load_json("knobs.json")
    if "_base" not in results:
        results["_base"] = {"val_loss": lolab.completion_loss(base, tok, evals), "text_loss": lolab.text_loss(base, tok, texts)}
        print(f"训练前：验证回复 loss {results['_base']['val_loss']:.4f}，普通文本 loss {results['_base']['text_loss']:.4f}")
    for name in names:
        desc, cfg, lr, opts = CONFIGS[name]
        lolab.banner(f"{name}：{desc}")
        lolab.seed(0)
        try:
            m, opt = build(tok, base, cfg, opts)
        except Exception as e:                       # QLoRA 需要 bitsandbytes（CPU 上还需要 kernels 包与联网取一次 kernel）
            print(f"  跳过：{type(e).__name__}: {str(e)[:200]}")
            continue
        _, trainable = lolab.count_params(m)
        state = lolab.training_state_bytes(trainable, total)
        print(f"  可训练参数 {trainable / 1e6:.1f}M（{trainable / total:.2%}），训练状态（BF16 冻结权重 + 可训练 16 B）{state / 2**30:.2f} GiB")
        losses, spt = lolab.run_sft(tok, m, train, steps, lr, name, optimizer=opt)
        val = lolab.completion_loss(m, tok, evals)
        txt = lolab.text_loss(m, tok, texts)
        row = {"desc": desc, "trainable": trainable, "lr": lr, "steps": steps, "sec_per_step": spt,
               "train_losses": losses, "val_loss": val, "text_loss": txt}
        results[name] = row
        print(f"  训练 loss {losses[0][1]:.3f} → {losses[-1][1]:.3f}   验证回复 loss {val:.4f}（训练前 {results['_base']['val_loss']:.4f}）"
              f"   普通文本 loss {txt:.4f}（{txt - results['_base']['text_loss']:+.4f}）   {spt:.1f} s/步")
        if not args.quick:
            ck = os.path.join(lolab.OUT, "ckpt")
            os.makedirs(ck, exist_ok=True)
            if opts.get("save"):
                m.save_pretrained(os.path.join(ck, name))              # 只存 adapter，03_deploy.py 用
            if name == "full":
                torch.save({k: v for k, v in m.state_dict().items() if v.ndim == 2 and "layers" in k}, os.path.join(ck, "full_linear.pt"))
            lolab.save_json("knobs.json", results)
        del m, opt
    lolab.banner("汇总")
    b = results["_base"]
    print(f"  {'配置':<34}{'可训练':>9}{'训练loss':>10}{'验证回复loss':>13}{'普通文本Δ':>11}{'s/步':>7}")
    for name in names:
        if name not in results:
            continue
        r = results[name]
        print(f"  {r['desc']:<34}{r['trainable'] / 1e6:>8.1f}M{r['train_losses'][-1][1]:>10.3f}{r['val_loss']:>13.4f}"
              f"{r['text_loss'] - b['text_loss']:>+11.4f}{r['sec_per_step']:>7.1f}")


if __name__ == "__main__":
    main()
