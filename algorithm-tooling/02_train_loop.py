"""PyTorch 使用层·上（工具箱 02）：二十行训练循环训一个字符级小 Transformer，loss 曲线正常下降。
https://arganzheng.life/pytorch-in-use-five-objects-and-a-training-loop.html

    python 02_train_loop.py            # 1000 步，CPU 约 1 分钟
    python 02_train_loop.py --quick    # 100 步
"""
import sys
import time

import torch
import torch.nn.functional as F

from tinygpt import Config, TinyGPT, cosine_with_warmup, get_batch, load_corpus, n_params

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main(quick=False):
    cfg = Config(steps=100 if quick else 1000, warmup=10 if quick else 50)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    train, val = load_corpus()
    print(f"语料: {len(train):,} 训练字符, {len(val):,} 验证字符; 词表 {cfg.vocab}; 初始 loss 应约 ln V = {torch.log(torch.tensor(float(cfg.vocab))):.2f}")

    # ---- 五个对象：Module / Optimizer / (Dataset+DataLoader 用 get_batch 代替) / Tensor / Autograd ----
    model = TinyGPT(cfg)
    print(f"模型: {cfg.layers} 层, d={cfg.d}, {n_params(model):,} 参数")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_with_warmup(s, cfg))

    # ---- 二十行训练循环 ----
    t0 = time.time()
    for step in range(cfg.steps):
        x, y = get_batch(train, cfg, gen)
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):   # CPU 没有 bf16 硬件，开了反而慢 30 倍
            logits = model(x)                                   # [B, T, V]
            loss = F.cross_entropy(logits.view(-1, cfg.vocab).float(), y.view(-1))
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if step % (20 if quick else 100) == 0 or step == cfg.steps - 1:
            print(f"step {step:4d}  loss {loss.item():.3f}  lr {sched.get_last_lr()[0]:.2e}  grad_norm {gnorm:.2f}  {time.time() - t0:5.1f}s")

    # ---- 评估：no_grad 下不建图 ----
    model.eval()
    with torch.no_grad():
        x, y = get_batch(val, Config(batch=64, seq=cfg.seq), gen)
        val_loss = F.cross_entropy(model(x).view(-1, cfg.vocab), y.view(-1)).item()
    print(f"验证 loss {val_loss:.3f}  (PPL {torch.exp(torch.tensor(val_loss)):.1f})")

    # ---- 生成几十个字符看看学到了什么 ----
    idx = torch.tensor([[ord(c) for c in "def "]])
    with torch.no_grad():
        for _ in range(80):
            logits = model(idx[:, -cfg.seq:])[:, -1] / 0.8
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
    print("采样(温度 0.8):", repr("".join(chr(i) for i in idx[0].tolist())))


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
