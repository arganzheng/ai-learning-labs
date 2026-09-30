"""LoRA 系列各脚本共用的小工具：设备、加载、数据、一次 SFT 的封装、两种 loss、参数与状态的账。
与 ../post-training/ptlab.py 同源，为了让本目录可以单独运行而复制一份。"""
import json
import math
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed(s=0):
    random.seed(s)
    torch.manual_seed(s)


def load(name=MODEL, dtype=torch.float32, **kw):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **kw)
    if "quantization_config" not in kw:
        model = model.to(device())
    return tok, model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def training_state_bytes(trainable, weights_total, weight_bytes=2):
    """混合精度 + AdamW：可训练参数各 4 B 主权重 + 8 B 两个矩 + 2 B 梯度；冻结权重只占 weight_bytes。"""
    return weights_total * weight_bytes + trainable * (4 + 8 + 2)


def lora_params(shapes, r):
    """shapes: [(d_in, d_out), ...]，返回 LoRA 参数量 sum r (d_in + d_out)。"""
    return sum(r * (i + o) for i, o in shapes)


# ---------------- 数据 ----------------
def load_data(n_train=800, n_eval=100):
    """no_robots：人写的一万条指令数据，取单轮、最后一条是 assistant 的样本。"""
    from datasets import load_dataset
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
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        paras = [t for t in ds["text"] if len(t) > 400 and not t.startswith(" =")]
        return paras[:n]
    except Exception:
        import sysconfig
        from pathlib import Path
        files = sorted(Path(sysconfig.get_paths()["stdlib"]).glob("*.py"))[:n]
        return [p.read_text(errors="ignore")[:2000] for p in files]


# ---------------- 训练封装 ----------------
def run_sft(tok, model, train, steps, lr, tag, bs=4, max_length=512, optimizer=None, completion_only=True):
    """trl 的 SFTTrainer 跑 steps 步，返回 [(step, loss)] 与每步秒数。optimizer 用于 LoRA+ 之类自定义 param group。"""
    from trl import SFTConfig, SFTTrainer
    from transformers.trainer_callback import PrinterCallback, ProgressCallback
    cfg = SFTConfig(
        output_dir=f"/tmp/lora-{tag}", max_steps=steps, per_device_train_batch_size=bs,
        gradient_accumulation_steps=1, learning_rate=lr, lr_scheduler_type="cosine", warmup_steps=max(1, steps // 10),
        weight_decay=0.0, logging_steps=max(1, steps // 8), save_strategy="no", report_to=[], seed=0, disable_tqdm=True,
        max_length=max_length, completion_only_loss=completion_only, packing=False, bf16=False, fp16=False,
        dataloader_pin_memory=False, use_cpu=(device() == "cpu"),
    )
    kw = {}
    if optimizer is not None:
        from transformers import get_cosine_schedule_with_warmup
        sched = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, steps)
        kw["optimizers"] = (optimizer, sched)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train, processing_class=tok, **kw)
    for cb in (PrinterCallback, ProgressCallback):
        trainer.remove_callback(cb)
    t = time.time()
    trainer.train()
    spt = (time.time() - t) / steps
    losses = [(h["step"], h["loss"]) for h in trainer.state.log_history if "loss" in h]
    return losses, spt


@torch.no_grad()
def completion_loss(model, tok, examples, max_length=768):
    """只在 assistant 回复的 token 上算平均交叉熵。"""
    model.eval()
    tot, n = 0.0, 0
    for ex in examples:
        p = _ids(tok.apply_chat_template(ex["prompt"], tokenize=True, add_generation_prompt=True))
        full = _ids(tok.apply_chat_template(ex["prompt"] + ex["completion"], tokenize=True))[:max_length]
        if len(full) <= len(p):
            continue
        labels = [-100] * len(p) + full[len(p):]
        x = torch.tensor([full], device=model.device)
        y = torch.tensor([labels], device=model.device)
        out = model(input_ids=x, labels=y)
        k = len(full) - len(p)
        tot += out.loss.item() * k
        n += k
    model.train()
    return tot / max(1, n)


def _ids(x):
    return x["input_ids"] if isinstance(x, dict) or hasattr(x, "keys") else list(x)


@torch.no_grad()
def text_loss(model, tok, texts, max_length=512):
    """普通文本上的平均交叉熵（衡量预训练能力是否被 SFT 冲掉）。"""
    model.eval()
    tot, n = 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(model.device)
        if ids.shape[1] < 2:
            continue
        out = model(input_ids=ids, labels=ids)
        tot += out.loss.item() * (ids.shape[1] - 1)
        n += ids.shape[1] - 1
    model.train()
    return tot / max(1, n)


def ppl(loss):
    return math.exp(loss)


def save_json(name, obj):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    return path


def load_json(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def banner(s):
    print(f"\n=== {s} ===", flush=True)
