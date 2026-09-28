"""第 8 步：评——val loss → 困惑度 → bits/byte（跨 tokenizer 可比），与 GPT-2 small（1.24 亿参数）在同一批
val 文档上比 bits/byte；看几条生成样本；试一个「聊天式」提示，看预训练模型为什么还不是助手。

    python step8_eval.py
"""
import contextlib
import io
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from common import DATA, fmt, read_jsonl
from trainer import BLOCK, DEV, make_model, n_params

os.environ.setdefault("HF_HUB_OFFLINE", "1")


@torch.no_grad()
def nll_on_docs(logprob_fn, docs, max_docs=40):
    """对每篇 val 文档算总 nats 与字节数，返回 (nats/token, bits/byte, tokens)。"""
    tot_nats = tot_tok = tot_bytes = 0
    for d in docs[:max_docs]:
        nats, ntok = logprob_fn(d["text"])
        tot_nats += nats; tot_tok += ntok; tot_bytes += len(d["text"].encode())
    return tot_nats / tot_tok, tot_nats / math.log(2) / tot_bytes, tot_tok


if __name__ == "__main__":
    tok = Tokenizer.from_file(f"{DATA}/tokenizer.json")
    ck = torch.load(f"{DATA}/ckpt.pt", map_location="cpu")
    with contextlib.redirect_stdout(io.StringIO()):
        model = make_model(ck["config"]["n_layer"], ck["config"]["n_embd"])
    model.load_state_dict(ck["model"]); model.eval()
    docs = read_jsonl(f"{DATA}/clean.jsonl")
    rng = np.random.default_rng(0)
    order = rng.permutation(len(docs))
    val_docs = [docs[i] for i in order[: max(1, int(len(docs) * 0.02))]]     # 与 step5 同一个划分
    print(f"模型：{ck['config']['n_layer']} 层 × {ck['config']['n_embd']} 宽，N = {n_params(model)/1e6:.2f}M；val {len(val_docs)} 篇文档")

    def ours(text):
        ids = tok.encode(text).ids
        nats = 0.0; n = 0
        for s in range(0, len(ids) - 1, BLOCK):
            chunk = torch.tensor(ids[s:s + BLOCK + 1], device=DEV)[None]
            if chunk.size(1) < 2:
                break
            logits, _ = model(chunk[:, :-1], chunk[:, 1:])
            lp = F.log_softmax(logits.float(), -1).gather(-1, chunk[:, 1:, None]).squeeze(-1)
            nats -= lp.sum().item(); n += lp.numel()
        return nats, n

    nat_tok, bpb, ntok = nll_on_docs(ours, val_docs)
    print(f"\n我们的模型：{nat_tok:.3f} nats/token（PPL {math.exp(nat_tok):.1f}），{bpb:.3f} bits/byte，共 {fmt(ntok)} token")
    print(f"  参照：随机猜 = ln 4096 = {math.log(4096):.2f} nats/token；训练前的 val loss 就是它")

    try:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        gtok = GPT2TokenizerFast.from_pretrained("gpt2")
        with contextlib.redirect_stdout(io.StringIO()):
            gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(DEV).eval()

        def gpt2_fn(text):
            ids = gtok(text).input_ids
            nats = 0.0; n = 0
            for s in range(0, len(ids) - 1, 1024):
                chunk = torch.tensor(ids[s:s + 1025], device=DEV)[None]
                if chunk.size(1) < 2:
                    break
                logits = gpt2(chunk[:, :-1]).logits
                lp = F.log_softmax(logits.float(), -1).gather(-1, chunk[:, 1:, None]).squeeze(-1)
                nats -= lp.sum().item(); n += lp.numel()
            return nats, n
        g_nat, g_bpb, g_tok = nll_on_docs(gpt2_fn, val_docs)
        print(f"GPT-2 small（124M，WebText 40 GB 训的）：{g_nat:.3f} nats/token（PPL {math.exp(g_nat):.1f}，它的 token 更长所以 PPL 不可直接比），{g_bpb:.3f} bits/byte，共 {fmt(g_tok)} token")
        print(f"  同一批文本，bits/byte 才可比：我们 {bpb:.3f} vs GPT-2 {g_bpb:.3f}——每个字节多花 {bpb-g_bpb:.2f} bit；参数少 {124/ (n_params(model, False)/1e6):.0f} 倍、数据少约 {40*1e9/ (sum(len(d['text'].encode()) for d in docs)):.0f} 倍")
    except Exception as e:  # noqa: BLE001
        print(f"（GPT-2 对照不可用：{e}）")

    torch.manual_seed(0)
    def gen(prompt, n=60, temp=0.8, top_k=40):
        ids = torch.tensor(tok.encode(prompt).ids, device=DEV)[None]
        out = model.generate(ids, n, temperature=temp, top_k=top_k)
        return tok.decode(out[0, ids.size(1):].tolist())
    print("\n生成样本（τ = 0.8，top-k 40）：")
    for p in ("The city council announced", "In this tutorial we will", "The best way to"):
        print(f"  {p!r} → {gen(p)!r}")
    print("\n试着当助手用：")
    for p in ("Q: What is the capital of France?\nA:", "Write a short poem about the sea.\n"):
        print(f"  {p!r}\n    → {gen(p, 50)!r}")
    print("\n它学到的是「网页文本的下一个 token 是什么」，不是「怎么回答问题」——这就是后训练要做的事（后训练系列第一篇）。")
