"""SFT 实验（后训练 01）：chat template、loss mask、padding 与 packing 的账、全量 vs LoRA、灾难性遗忘。
https://arganzheng.life/sft-data-chat-template-loss-mask-and-peft.html

模型 Qwen/Qwen2.5-0.5B（base），数据 HuggingFaceH4/no_robots（人写的 1 万条指令数据）。
Apple Silicon（MPS）或 CUDA 上完整运行约 40 分钟；CPU 慢 5–10 倍，建议 --quick。

    python 01_sft.py                       # 全部：template padding mask lora forget
    python 01_sft.py template padding      # 不训练的两个，几秒
    python 01_sft.py mask --quick          # 训练类实验缩短到十几步
"""
import argparse
import copy
import random
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

import ptlab

MODEL = "Qwen/Qwen2.5-0.5B"
random.seed(0)
torch.manual_seed(0)


# ---------------- 数据 ----------------
def load_data(n_train=800, n_eval=100):
    ds = load_dataset("HuggingFaceH4/no_robots")

    def to_pc(ex):
        msgs = ex["messages"]
        return {"prompt": msgs[:-1], "completion": msgs[-1:]}

    def ok(ex):
        return ex["messages"][-1]["role"] == "assistant" and len(ex["messages"]) <= 3

    train = ds["train"].filter(ok).select(range(n_train)).map(to_pc, remove_columns=ds["train"].column_names)
    evals = ds["test"].filter(ok).select(range(n_eval)).map(to_pc, remove_columns=ds["test"].column_names)
    return train, [dict(e) for e in evals]


def load_plain_text(n=60):
    """衡量遗忘用的普通文本：wikitext-2 的测试段落；没网络时退回 Python 标准库源码。"""
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        paras = [t for t in ds["text"] if len(t) > 400 and not t.startswith(" =")]
        return paras[:n]
    except Exception:
        import sysconfig
        from pathlib import Path
        files = sorted(Path(sysconfig.get_paths()["stdlib"]).glob("*.py"))[:n]
        return [p.read_text(errors="ignore")[:2000] for p in files]


# ---------------- 实验 1：chat template ----------------
def exp_template(tok, model):
    print("=== 实验 1：chat template 把对话变成什么 ===")
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4."},
            {"role": "user", "content": "And 3+3?"}]
    rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print(rendered)
    ids = tok(rendered).input_ids
    specials = [(i, tok.convert_ids_to_tokens(t)) for i, t in enumerate(ids) if t in tok.all_special_ids]
    print(f"  共 {len(ids)} 个 token，其中特殊 token {len(specials)} 个：{[s for _, s in specials]}")
    print(f"  im_start / im_end 的 id：{tok.convert_tokens_to_ids('<|im_start|>')} / {tok.convert_tokens_to_ids('<|im_end|>')}"
          f"（tokenizer 共 {len(tok)} 个 token，其中基础词表 {tok.vocab_size}；模型 embedding 有 {model.config.vocab_size} 行，多出的是预留位）")
    naive = "System: You are a helpful assistant.\nUser: What is 2+2?\nAssistant: 4.\nUser: And 3+3?\nAssistant:"
    print(f"  同一段对话不用模板、手拼 'User:/Assistant:'：{len(tok(naive).input_ids)} 个 token，没有任何特殊 token——"
          "训练时用这种格式、推理时用官方模板（或反过来），模型看到的是两种没见过对方的分布。\n")


# ---------------- 实验 2：padding 与 packing 的账 ----------------
def exp_padding(tok, train):
    print("=== 实验 2：不 packing 时 padding 浪费多少 ===")
    lens = [len(ptlab._ids(tok.apply_chat_template(ex["prompt"] + ex["completion"], tokenize=True))) for ex in train]
    plens = sorted(len(ptlab._ids(tok.apply_chat_template(ex["prompt"], tokenize=True, add_generation_prompt=True))) for ex in train)
    lens.sort()
    n = len(lens)
    print(f"  {n} 条样本的 token 数：中位数 {lens[n // 2]}，p90 {lens[int(n * 0.9)]}，最大 {lens[-1]}，合计 {sum(lens):,}")
    print(f"  其中 prompt 部分（含模板与 system）中位数 {plens[n // 2]}，回复部分中位数约 {lens[n // 2] - plens[n // 2]}："
          f"不 mask 时约 {sum(plens) / sum(lens):.0%} 的 loss 位置落在 prompt 上")
    for L in [512, 1024, 2048]:
        clipped = [min(l, L) for l in lens]
        for bs in [8]:
            random.seed(0)
            order = clipped[:]
            random.shuffle(order)
            padded = sum(max(order[i : i + bs]) * len(order[i : i + bs]) for i in range(0, n, bs))
            packed_seqs = -(-sum(clipped) // L)
            print(f"  max_length {L:>4}，batch {bs}：动态 padding 后实际算了 {padded:>9,} 个位置，有效 {sum(clipped) / padded:.0%}；"
                  f"packing 只需 {packed_seqs} 条 {L} 的序列（有效 {sum(clipped) / (packed_seqs * L):.0%}）")
    print("  长度按相近分组（length grouping）能把动态 padding 的浪费降到 10% 以内；packing 接近 100%，"
          "但要配 attention 掩码（或 padding-free 的变长 kernel）防止跨样本 attention。\n")


# ---------------- 训练封装 ----------------
def run_sft(tok, model, train, steps, lr, completion_only, tag, bs=4, max_length=512):
    cfg = SFTConfig(
        output_dir=f"/tmp/sft-{tag}", max_steps=steps, per_device_train_batch_size=bs,
        gradient_accumulation_steps=1, learning_rate=lr, lr_scheduler_type="cosine", warmup_steps=max(1, steps // 10),
        weight_decay=0.0, logging_steps=max(1, steps // 8), save_strategy="no", report_to=[], seed=0, disable_tqdm=True,
        max_length=max_length, completion_only_loss=completion_only, packing=False, bf16=False, fp16=False,
        dataloader_pin_memory=False,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train, processing_class=tok)
    from transformers.trainer_callback import PrinterCallback, ProgressCallback
    for cb in (PrinterCallback, ProgressCallback):
        trainer.remove_callback(cb)
    with ptlab.Timer() as t:
        trainer.train()
    losses = [(h["step"], h["loss"]) for h in trainer.state.log_history if "loss" in h]
    return losses, t.s / steps


# ---------------- 实验 3：loss mask ----------------
def exp_mask(tok, base, train, evals, steps):
    print(f"=== 实验 3：loss 算不算 prompt（completion_only_loss），各训 {steps} 步，lr 1e-5，全量 ===")
    before = ptlab.completion_loss(base, tok, evals)
    print(f"  训练前 base 模型在验证集回复上的 loss：{before:.3f}（PPL {ptlab.ppl(before):.1f}）")
    for mask in [True, False]:
        model = copy.deepcopy(base)
        losses, spt = run_sft(tok, model, train, steps, 1e-5, mask, f"mask{mask}")
        after = ptlab.completion_loss(model, tok, evals)
        print(f"  completion_only_loss={mask!s:<5} 训练 loss {losses[0][1]:.3f} → {losses[-1][1]:.3f}   "
              f"验证回复 loss {before:.4f} → {after:.4f}   {spt:.1f} s/步")
        del model
    print("  看什么：不 mask 时训练 loss 的数值本身就不同（混进了 prompt 的 token，不能与 mask 的 loss 直接比）；"
          "验证集只算回复，两者的差才是 mask 的真实影响。prompt 短、回复长的数据（本数据集 prompt 只占 34%）里"
          "两者接近甚至不 mask 略好——prompt 的文字是额外的语言建模信号；prompt 占大头的数据上 mask 的优势才明显。\n")


# ---------------- 实验 4：全量 vs LoRA ----------------
def lora_variants():
    attn = ["q_proj", "k_proj", "v_proj", "o_proj"]
    allp = attn + ["gate_proj", "up_proj", "down_proj"]
    return [("LoRA r=16 attention", LoraConfig(r=16, lora_alpha=32, target_modules=attn, task_type="CAUSAL_LM")),
            ("LoRA r=16 全部线性层", LoraConfig(r=16, lora_alpha=32, target_modules=allp, task_type="CAUSAL_LM")),
            ("LoRA r=64 全部线性层", LoraConfig(r=64, lora_alpha=128, target_modules=allp, task_type="CAUSAL_LM"))]


def exp_lora(tok, base, train, evals, steps):
    print(f"=== 实验 4：全量微调 vs LoRA——参数、训练状态、{steps} 步后的验证 loss ===")
    total, _ = ptlab.count_params(base)
    rows = [("全量", None)] + lora_variants()
    print(f"  {'配置':<22}{'可训练参数':>12}{'占比':>8}{'训练状态(混合精度)':>18}{'8B 规格同比例':>14}")
    for name, cfg in rows:
        m = copy.deepcopy(base) if cfg is None else get_peft_model(copy.deepcopy(base), cfg)
        _, trainable = ptlab.count_params(m)
        state = ptlab.training_state_bytes(trainable, total)
        state8b = ptlab.training_state_bytes(int(trainable / total * 8.03e9), 8.03e9)
        print(f"  {name:<22}{trainable / 1e6:>10.1f}M{trainable / total:>8.2%}{state / 2**30:>14.2f} GiB{state8b / 2**30:>11.0f} GiB")
        del m
    print("  （训练状态 = 冻结权重 BF16 2 B + 可训练参数的 FP32 主权重 4 B + Adam 8 B + 梯度 2 B；不含激活值）\n")
    results = []
    for name, cfg, lr in [("全量 lr 1e-5", None, 1e-5), ("LoRA r=16 全部线性层 lr 1e-4", lora_variants()[1][1], 1e-4)]:
        m = copy.deepcopy(base) if cfg is None else get_peft_model(copy.deepcopy(base), cfg)
        losses, spt = run_sft(tok, m, train, steps, lr, True, name.replace(" ", ""))
        after = ptlab.completion_loss(m, tok, evals)
        results.append((name, losses[-1][1], after, spt))
        del m
    for name, tl, vl, spt in results:
        print(f"  {name:<32} 训练 loss {tl:.3f}   验证回复 loss {vl:.4f}   {spt:.1f} s/步")
    print("  看什么：LoRA 的可训练参数不到 2%，训练状态降到全量的几分之一；同样步数下验证 loss 略高，"
          "lr 要比全量大一个数量级（LoRA 的有效更新幅度被 alpha/r 与低秩结构缩放）。\n")


# ---------------- 实验 5：灾难性遗忘 ----------------
def exp_forget(tok, base, train, evals, steps):
    print(f"=== 实验 5：SFT 之后预训练能力掉多少——普通文本上的 loss（{steps} 步）===")
    texts = load_plain_text()
    before_text = ptlab.text_loss(base, tok, texts)
    before_chat = ptlab.completion_loss(base, tok, evals)
    print(f"  训练前：普通文本 loss {before_text:.4f}（PPL {ptlab.ppl(before_text):.1f}），验证回复 loss {before_chat:.4f}")
    print(f"  {'配置':<28}{'普通文本 loss':>14}{'变化':>8}{'验证回复 loss':>14}")
    for name, cfg, lr in [("全量 lr 1e-5", None, 1e-5), ("全量 lr 1e-4", None, 1e-4),
                          ("LoRA r=16 全部线性层 lr 1e-4", lora_variants()[1][1], 1e-4)]:
        m = copy.deepcopy(base) if cfg is None else get_peft_model(copy.deepcopy(base), cfg)
        run_sft(tok, m, train, steps, lr, True, "forget" + name.replace(" ", ""))
        t, c = ptlab.text_loss(m, tok, texts), ptlab.completion_loss(m, tok, evals)
        print(f"  {name:<28}{t:>14.4f}{t - before_text:>+8.4f}{c:>14.4f}")
        del m
    print("  看什么：lr 大一个数量级，回复 loss 降得更快，普通文本的 loss 也涨得更多——这就是遗忘；"
          "LoRA 因为只动低秩子空间，对普通文本的扰动通常更小。真实配方用 1e-5 量级、混入预训练数据回放、或对 checkpoint 做平均。\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="*", choices=["template", "padding", "mask", "lora", "forget"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    exps = args.experiments or ["template", "padding", "mask", "lora", "forget"]
    steps = 12 if args.quick else 80
    print(f"设备 {ptlab.device()}，模型 {MODEL}，训练类实验 {steps} 步（batch 4 × max_length 512）\n")
    tok, base = ptlab.load(MODEL)
    train, evals = load_data()
    t0 = time.time()
    if "template" in exps:
        exp_template(tok, base)
    if "padding" in exps:
        exp_padding(tok, train)
    if "mask" in exps:
        exp_mask(tok, base, train, evals, steps)
    if "lora" in exps:
        exp_lora(tok, base, train, evals, steps)
    if "forget" in exps:
        exp_forget(tok, base, train, evals, steps)
    print(f"总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
