"""GPU 直觉与实验管理（工具箱 05）：两个上限的算术、torch.profiler 看一步训练花在哪、一次实验的最小记录与 seed 复现。
https://arganzheng.life/gpu-intuition-and-experiment-management.html

    python 05_profiler_and_record.py            # 全部：roofline profile record
    python 05_profiler_and_record.py profile
"""
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from tinygpt import Config, TinyGPT, get_batch, load_corpus, n_params

OUT = Path(__file__).with_name("out")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- 1. 两个上限：纯算术 ----------------
def exp_roofline():
    print("=== 1. 两个上限（H100 SXM: 989 TFLOPS bf16, 3.35 TB/s）===")
    flops, bw = 989e12, 3.35e12
    print(f"  ridge = {flops/bw:.0f} FLOP/字节：算术强度高于它受算力限制，低于它受带宽限制")
    N = 8.03e9
    print(f"  decode（batch 1）: 读 {N*2/1e9:.1f} GB 权重做 {2*N/1e9:.1f} GFLOP → 强度 ≈ 1 → 时间下限 {N*2/bw*1e3:.1f} ms/token ≈ {bw/(N*2):.0f} token/s")
    for b in (1, 32, 128, 512):
        t_mem = N * 2 / bw
        t_comp = 2 * N * b / flops
        print(f"    batch {b:>4}: 带宽时间 {t_mem*1e3:.1f} ms, 算力时间 {t_comp*1e3:.1f} ms → {'memory' if t_mem > t_comp else 'compute'}-bound, "
              f"{b/max(t_mem, t_comp):.0f} token/s")
    T = 4096
    print(f"  prefill {T} token: {2*N*T/1e12:.0f} TFLOP → {2*N*T/flops*1e3:.0f} ms；强度是 decode 的 {T} 倍 → compute-bound")
    print()


# ---------------- 2. profiler：一步训练花在哪 ----------------
def one_step(model, x, y, opt):
    with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
        loss = F.cross_entropy(model(x).view(-1, model.cfg.vocab).float(), y.view(-1))
    loss.backward()
    opt.step(); opt.zero_grad(set_to_none=True)
    return loss


def exp_profile():
    from torch.profiler import ProfilerActivity, profile

    print("=== 2. torch.profiler：一步训练的前几个算子 ===")
    cfg = Config()
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    train, _ = load_corpus(300_000)
    model = TinyGPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    x, y = get_batch(train, cfg, gen)
    for _ in range(3):
        one_step(model, x, y, opt)                                  # 预热
    t0 = time.time(); one_step(model, x, y, opt); wall = time.time() - t0
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        one_step(model, x, y, opt)
    print(f"  模型 {n_params(model)/1e6:.2f} M 参数, batch {cfg.batch} × seq {cfg.seq}; 一步 {wall*1e3:.0f} ms（CPU）")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=8, max_name_column_width=40))
    total = sum(e.self_cpu_time_total for e in prof.key_averages())
    top = sorted(prof.key_averages(), key=lambda e: -e.self_cpu_time_total)[:3]
    print("  前三个算子占比:", ", ".join(f"{e.key} {e.self_cpu_time_total/total*100:.0f}%" for e in top))
    print("  读法：矩阵乘（mm / addmm / bmm）与 attention 应占大头；如果 copy_ / to / 小算子占大头，就是形状转换或 launch 开销在吃时间")
    print()


# ---------------- 3. 实验记录与 seed 复现 ----------------
def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "n/a"


def run(cfg, steps=20):
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    train, _ = load_corpus(300_000)
    model = TinyGPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    for _ in range(steps):
        x, y = get_batch(train, cfg, gen)
        loss = one_step(model, x, y, opt)
    return loss.item()


def exp_record():
    print("=== 3. 一次实验的最小记录 + seed 复现 ===")
    cfg = Config(steps=20)
    losses = {}
    for seed in (0, 0, 1):
        c = Config(**{**asdict(cfg), "seed": seed})
        losses.setdefault(seed, []).append(run(c))
    print(f"  seed 0 两次: {losses[0][0]:.6f} vs {losses[0][1]:.6f} → {'一致' if losses[0][0] == losses[0][1] else '不一致（CPU 上通常一致；GPU 上某些 kernel 非确定）'}")
    print(f"  seed 1:      {losses[1][0]:.6f} → 与 seed 0 差 {abs(losses[1][0]-losses[0][0]):.4f}，这就是'单个数字不算结论'的原因")
    record = {
        "run_id": time.strftime("%Y%m%d-%H%M%S"),
        "commit": git_commit(),
        "config": asdict(cfg),
        "data": {"source": "python stdlib .py", "sha256_prefix": hashlib.sha256(str(load_corpus(300_000)[0][:1000].tolist()).encode()).hexdigest()[:12]},
        "env": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform()},
        "metrics": {"loss_seed0": losses[0][0], "loss_seed1": losses[1][0]},
    }
    OUT.mkdir(exist_ok=True)
    path = OUT / f"run-{record['run_id']}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"  记录写到 {path}：run id · commit · 配置 · 数据版本 · seed · 环境 · 指标 —— 七项齐了才谈复现")
    print("  " + json.dumps({k: record[k] for k in ("run_id", "commit", "metrics")}))


EXPS = {"roofline": exp_roofline, "profile": exp_profile, "record": exp_record}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(EXPS)
    for n in names:
        EXPS[n]()
