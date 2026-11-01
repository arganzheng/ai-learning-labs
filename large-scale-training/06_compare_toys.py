"""06 三框架对比 toy：同一个小模型在四种分片表示下的切法，与同一个 1F1B 的三种调度表示。

    python 06_compare_toys.py          # 全部实验
    python 06_compare_toys.py shard    # A[4,3] B[2,3] c[5] 共 23 元素，N_d=2 的四种切法
    python 06_compare_toys.py pp       # PP=2、m=4，三个框架各打印一遍 rank0 的 1F1B
    python 06_compare_toys.py saw      # 16 层、预取深度 1 的 gather/compute/release 事件序列（喂给 06 图 F3）

shard 与 fsdp2 部分需要 2 个进程（gloo）；pp 与 saw 是单进程打印。
"""
from __future__ import annotations

import argparse
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 2

# 同一个小模型的三个参数：名字、形状、元素编号区间（拍平后）
PARAMS = [("A", (4, 3)), ("B", (2, 3)), ("c", (5,))]
NUMEL = sum(torch.Size(s).numel() for _, s in PARAMS)   # 12 + 8 + 5 = 23


def _p(rank, *args):
    print(f"[rank{rank}] " + " ".join(str(a) for a in args))


def _intervals_in(start, end, params_flat):
    """[start,end) 覆盖了哪些参数的哪个元素区间。params_flat = [(name, off, numel)]。"""
    out = []
    for name, off, n in params_flat:
        s, e = max(start, off), min(end, off + n)
        if s < e:
            out.append(f"{name}[{s - off}:{e - off}]")
    return out or ["（无）"]


def _shard(rank):
    torch.manual_seed(0)
    tensors = [torch.randn(s) for _, s in PARAMS]
    flats = [t.reshape(-1) for t in tensors]
    # 声明顺序的拍平偏移：A[0:12] B[12:18] c[18:23]
    decl, _off = [], 0
    for (n, _), f in zip(PARAMS, flats):
        decl.append((n, _off, f.numel()))
        _off += f.numel()

    _p(rank, f"参数 A[4,3] B[2,3] c[5]，共 {NUMEL} 个元素，N_d={WORLD}")

    # a. Megatron 分布式优化器：所有参数拍进一段连续 buffer（按声明逆序），
    #    pad 到 N_d 整除，按字节区间均分；区间可以跨参数边界。
    _p(rank, "-- a. Megatron 分布式优化器（一段连续 buffer + pad 到 N_d 倍数）")
    rev = list(reversed(flats))          # c, B, A —— 03 toy 的 buffer 内偏移同款逆序
    rev_names = list(reversed([n for n, _ in PARAMS]))
    rev_off, off = [], 0
    for n, f in zip(rev_names, rev):
        rev_off.append((n, off, f.numel()))
        off += f.numel()
    pad = (-NUMEL) % WORLD
    buf = torch.cat(rev + [torch.zeros(pad)])
    shard = NUMEL + pad
    half = shard // WORLD
    s0, e0 = rank * half, (rank + 1) * half
    _p(rank, f"buffer 布局（逆序拍平）：{[(n, o, o + l) for n, o, l in rev_off]}，pad={pad}")
    _p(rank, f"本 rank 区间 [{s0},{e0})：{_intervals_in(s0, e0, rev_off)}"
       + (f"，外加 {e0 - NUMEL} 个 pad 元素" if e0 > NUMEL else ""))

    # b. DeepSpeed ZeRO-3：每个参数各自拍平、pad 到 N_d 整除、切成 N_d 段（ds_tensor）。
    _p(rank, "-- b. DeepSpeed ZeRO-3（每参数各自 pad + 均分，ds_tensor）")
    for (n, _), f in zip(PARAMS, flats):
        n_ = f.numel()
        seg = -(-n_ // WORLD)             # ceil 切长，含尾部 padding
        lo, hi = rank * seg, min((rank + 1) * seg, n_)
        real = max(0, hi - lo)
        _p(rank, f"{n}: numel={n_} pad={seg * WORLD - n_} 段长={seg}，本 rank 拿 {n}[{lo}:{lo + real}]"
           + (f"（其余 {seg - real} 个为 padding）" if seg - real else ""))

    # c. FSDP1：一个 wrap 单元的所有参数拍平拼成 FlatParameter，pad，均分。
    _p(rank, "-- c. FSDP1（整单元拍平成 FlatParameter + pad + 均分）")
    flat = torch.cat(flats)
    shard1 = -(-NUMEL // WORLD)
    pad1 = shard1 * WORLD - NUMEL
    s1, e1 = rank * shard1, min((rank + 1) * shard1, NUMEL)
    _p(rank, f"FlatParameter numel={NUMEL} pad={pad1}，本 rank [{s1},{e1})：{_intervals_in(s1, e1, decl)}"
       + ("（尾部含 padding）" if e1 >= NUMEL else ""))

    # d. FSDP2：每个参数用真实 distribute_tensor(Shard(0)) 切第 0 维。
    _p(rank, "-- d. FSDP2（真实 DTensor，distribute_tensor Shard(0)）")
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, Shard
    mesh = init_device_mesh("cpu", (WORLD,))
    for (n, shp), t in zip(PARAMS, tensors):
        d = distribute_tensor(t.clone(), mesh, [Shard(0)])
        _p(rank, f"{n}{shp}: 全局形状 {tuple(d.shape)}，本 rank to_local()={tuple(d.to_local().shape)}")


def _schedule(pp, m, stage):
    """非交错 1F1B（03 toy 同款）：warmup = pp - stage - 1 个 F，然后 F/B 交替，最后补齐 B。"""
    n_warm = pp - stage - 1
    seq = [("F", i) for i in range(min(n_warm, m))]
    f_i, b_i = n_warm, 0
    while b_i < m:
        if f_i < m:
            seq.append(("F", f_i))
            f_i += 1
        seq.append(("B", b_i))
        b_i += 1
    return seq


def _pp(_=None):
    pp, m = 2, 4
    print(f"PP={pp}、每个 stage m={m} 个 micro-batch，只看 rank0（stage0）")

    print("-- Megatron：过程式三段循环产生的序列")
    for st in range(pp):
        seq = _schedule(pp, m, st)
        warm = [f"{k}{i + 1}" for k, i in seq if k == "F" and i < pp - st - 1]
        cool_i = len([1 for k, _ in seq if k == "B"]) - (m - min(pp - st - 1, m))
        warm_n = min(pp - st - 1, m)
        warm = [f"F{i + 1}" for i in range(warm_n)]
        cool = [f"B{i + 1}" for i in range(m - warm_n, m)]
        steady = [f"{k}{i + 1}" for k, i in seq[warm_n:len(seq) - len(cool)]]
        print(f"  stage{st}: warmup {warm} | steady {steady} | cooldown {cool}")
        print(f"          序列: {' '.join(f'{k}{i + 1}' for k, i in seq)}")

    print("-- DeepSpeed：TrainSchedule 的 stage0 指令流（按 v0.19.2 schedule.py 的 steps() 生成）")
    commands = []
    prev_micro = -1
    for step in range(2 * (m + pp - 1)):
        if step % 2 == 0:
            micro, forward = step // 2, True
        else:
            micro, forward = (step - 1) // 2 - pp + 1, False
        if forward:
            if 0 <= prev_micro < m:
                commands.append(f"SendGrad({prev_micro})")
            if 0 <= micro < m:
                commands.extend((f"LoadMicroBatch({micro})", f"ForwardPass({micro})"))
        else:
            if 0 <= micro < m:
                commands.append(f"RecvGrad({micro})")
            if 0 <= prev_micro < m:
                commands.append(f"SendActivation({prev_micro})")
            if 0 <= micro < m:
                commands.append(f"BackwardPass({micro})")
        if step == 2 * (m + pp - 1) - 1:
            commands.extend(("ReduceTiedGrads()", "ReduceGrads()", "OptimizerStep()"))
        prev_micro = micro
    for i in range(0, len(commands), 4):
        print("   " + "  ".join(f"{x:<23}" for x in commands[i:i + 4]))

    print("-- PyTorch：真实 Schedule1F1B._get_pipeline_order + _format_pipeline_order（CPU 可生成）")
    from types import SimpleNamespace
    from torch.distributed.pipelining.schedules import Schedule1F1B, _format_pipeline_order
    schedule = Schedule1F1B(SimpleNamespace(num_stages=pp), n_microbatches=m)
    print(_format_pipeline_order(schedule.pipeline_order))


def saw_events(n_layers, prefetch):
    """照 04 toy memory 实验的语义生成 16 层事件序列：
    每个事件 (direction, op, layer, live_layers)。预取=1 时峰值 2 层同时在。"""
    ev = []

    def run(direction, depth):
        order = range(n_layers) if direction == "forward" else range(n_layers - 1, -1, -1)
        live, gathered = 0, set()
        for i in order:
            if i not in gathered:
                ev.append((direction, "gather", i, live + 1))
                live += 1
                gathered.add(i)
            nxt = i + 1 if direction == "forward" else i - 1
            if depth and 0 <= nxt < n_layers and nxt not in gathered:
                ev.append((direction, "prefetch", nxt, live + 1))
                live += 1
                gathered.add(nxt)
            ev.append((direction, "compute", i, live))
            ev.append((direction, "release", i, live - 1))
            live -= 1
    run("forward", prefetch)
    run("backward", prefetch)
    return ev


def _saw(_=None):
    ev = saw_events(16, prefetch=1)
    for d, op, i, live in ev:
        print(f"  {d} {op:<9} layer{i:<2} live_full_layers={live}")
    print(f"  peak={max(l for _, _, _, l in ev)} layers")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _shard_entry(rank, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    try:
        _shard(rank)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", nargs="*", choices=("shard", "pp", "saw"))
    args = ap.parse_args()
    for ex in args.experiment or ("shard", "pp", "saw"):
        print(f"== {ex} ==")
        if ex == "shard":
            mp.spawn(_shard_entry, args=(_free_port(),), nprocs=WORLD, join=True)
        elif ex == "pp":
            _pp()
        else:
            _saw()
