"""后训练系列各脚本共用的小工具：设备选择、加载模型、两种 loss 的评估、参数与状态的账。"""
import math
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(name, dtype=torch.float32):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device())
    return tok, model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def training_state_bytes(trainable, weights_total, weight_bytes=2, master=True):
    """混合精度 + AdamW：可训练参数各 4 B 主权重 + 8 B 两个矩 + 2 B 梯度；冻结权重只占 weight_bytes。"""
    per_trainable = 8 + 2 + (4 if master else 0)
    return weights_total * weight_bytes + trainable * per_trainable


@torch.no_grad()
def completion_loss(model, tok, examples, max_length=768):
    """只在 assistant 回复的 token 上算平均交叉熵。examples: [{'prompt': [...msgs], 'completion': [...msgs]}]
    逐条计算：logits 是 [T, V] 的 FP32 张量（768 × 152K ≈ 470 MB），批量算在小显存设备上反而更慢。"""
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
        out = model(input_ids=x, labels=y)          # transformers 内部做 shift 并忽略 -100
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


class Timer:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        self.s = time.time() - self.t


def ppl(loss):
    return math.exp(loss)
