"""大规模训练工程（05）torchtitan — 用 CPU/gloo rank 演示 FSDP2 的原语：DeviceMesh、DTensor、meta 构造与 fully_shard。

博客：https://arganzheng.life/torchtitan-architecture-and-source-guide.html

    python 05_torchtitan_toys.py                 # 全部
    python 05_torchtitan_toys.py mesh             # 8 rank，FSDP 4 × TP 2 的 DeviceMesh 分组
    python 05_torchtitan_toys.py dtensor          # 2 rank，Shard / Replicate / Partial 与 redistribute
    python 05_torchtitan_toys.py meta             # 2 rank，meta 构造 → fully_shard → to_empty → init → optimizer
    python 05_torchtitan_toys.py fsdp             # 4 rank，hook 记录 + 一次训练步，对照单进程参考

Toy 使用 CPU float32 参数、bf16 的 MixedPrecisionPolicy，聚焦抽象的用法与通信顺序。
"""
import argparse
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _init(rank, world, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)


# ---------------------------------------------------------------- mesh

def _mesh(rank):
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cpu", (4, 2), mesh_dim_names=("dp_shard", "tp"))
    coord = mesh.get_coordinate()
    coords = [torch.zeros(2, dtype=torch.long) for _ in range(8)]
    dist.all_gather(coords, torch.tensor(coord, dtype=torch.long))
    if rank == 0:
        for r, c in enumerate(coords):
            print(f"  rank {r}: mesh 坐标 dp_shard={int(c[0])} tp={int(c[1])}")
        tp_group = mesh["tp"].get_group()
        dp_group = mesh["dp_shard"].get_group()
        tp_ranks = sorted(dist.get_process_group_ranks(tp_group))
        dp_ranks = sorted(dist.get_process_group_ranks(dp_group))
        print(f"  rank 0 所在组: tp {tp_ranks}，dp_shard {dp_ranks}")
        print("  全部组（mesh[i][j] = rank，行是 dp_shard、列是 tp，TP 在最内维）:")
        for i in range(4):
            row = [int(mesh.mesh[i, j]) for j in range(2)]
            print(f"    dp_shard={i}: tp 组 {row}")
        print("    → dp_shard 组: [0,2,4,6] / [1,3,5,7]（跨 tp 列取同列）")
    dist.barrier()


# ---------------------------------------------------------------- dtensor

def _dtensor(rank):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, Replicate, Partial, distribute_tensor

    mesh = init_device_mesh("cpu", (2,))
    w = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    w_shard = distribute_tensor(w, mesh, placements=[Shard(0)])
    w_rep = distribute_tensor(w, mesh, placements=[Replicate()])
    # Partial 只能由 Shard 转来：每 rank 的部分和是自己的分片（其余位置为 0）
    partial = w_shard.redistribute(placements=[Partial()])

    shard_locals = [torch.empty_like(w_shard.to_local()) for _ in range(2)]
    dist.all_gather(shard_locals, w_shard.to_local().contiguous())
    if rank == 0:
        print("  W.shape=(4, 4)，元素 0–15 按行编号")
        for r, local in enumerate(shard_locals):
            print(f"  rank{r} Shard(0) local = {local.tolist()}")
        print(f"  Replicate local shape = {tuple(w_rep.to_local().shape)}（每 rank 完整一份）")
        print(f"  Partial local：rank0 前两行是自己的分片、其余为 0；rank1 则相反")

    full_from_shard = w_shard.redistribute(placements=[Replicate()])
    partial_sum = partial.redistribute(placements=[Replicate()])
    ok_gather = torch.equal(full_from_shard.to_local(), w)
    ok_all_reduce = torch.equal(partial_sum.to_local(), w)
    # Partial→Shard(0) 的语义就是 reduce-scatter：求和后每 rank 取回自己那 2×4 一段。
    back_to_shard = partial.redistribute(placements=[Shard(0)])
    ok_reduce_scatter = torch.equal(back_to_shard.to_local(), w_shard.to_local())
    ok_full = torch.equal(w_shard.full_tensor(), w)  # 全 rank 参与的 all-gather
    if rank == 0:
        print(f"  Shard(0)→Replicate = all-gather，full == W：{'OK' if ok_gather else 'FAIL'}")
        print(f"  Partial→Replicate = all-reduce，两份部分和拼回 W：{'OK' if ok_all_reduce else 'FAIL'}")
        print(f"  Partial→Shard(0) = reduce-scatter，与 Shard 分片一致："
              f"{'OK' if ok_reduce_scatter else 'FAIL'}")
        print(f"  Shard(0).full_tensor() == 原 W：{'OK' if ok_full else 'FAIL'}")


# ---------------------------------------------------------------- meta

class _Block(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.attn = nn.Linear(d, d, bias=False)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d, bias=False), nn.Linear(4 * d, d, bias=False))

    def forward(self, x):
        return self.ffn(self.attn(x))


class _Tiny(nn.Module):
    def __init__(self, d=16, n=3):
        super().__init__()
        self.layers = nn.ModuleList(_Block(d) for _ in range(n))
        self.head = nn.Linear(d, d, bias=False)

    def forward(self, x):
        for b in self.layers:
            x = b(x)
        return self.head(x)


def _param_bytes(model):
    total = 0
    for p in model.parameters():
        local = p.to_local() if hasattr(p, "to_local") else p
        if local.device.type != "meta":
            total += local.numel() * local.element_size()
    return total


def _meta(rank):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    mesh = init_device_mesh("cpu", (2,))
    torch.manual_seed(606)

    def report(stage):
        ranks_bytes = [torch.zeros(1, dtype=torch.long) for _ in range(2)]
        dist.all_gather(ranks_bytes, torch.tensor([_param_bytes(model)], dtype=torch.long))
        if rank == 0:
            print(f"  {stage}: 每 rank 实存参数 {[int(v[0]) for v in ranks_bytes]} B")

    with torch.device("meta"):
        model = _Tiny()
    if rank == 0:
        n_meta = sum(p.numel() for p in model.parameters())
        print(f"  meta 构造后: {n_meta} 个逻辑参数元素，实存 {_param_bytes(model)} B（只有形状）")

    for block in model.layers:
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)
    report("fully_shard 后（仍是 meta）")

    model.to_empty(device="cpu")
    report("to_empty 后（分片形状的空壳）")

    model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
    report("init 后（分片有真实值）")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(2, 16)
    loss = model(x).sum()
    loss.backward()
    opt.step()
    opt_bytes = sum(
        sum(st[key].numel() * st[key].element_size() for key in ("exp_avg", "exp_avg_sq") if key in st)
        for st in opt.state.values()
    )
    ranks_bytes = [torch.zeros(1, dtype=torch.long) for _ in range(2)]
    dist.all_gather(ranks_bytes, torch.tensor([opt_bytes], dtype=torch.long))
    if rank == 0:
        print(f"  optimizer.step 后: 每 rank m+v {[int(v[0]) for v in ranks_bytes]} B（fp32 分片上的 AdamW）")


# ---------------------------------------------------------------- fsdp

def _fsdp(rank):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

    world = 4
    mesh = init_device_mesh("cpu", (world,))
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    torch.manual_seed(707)
    model = _Tiny(d=16, n=3)
    for block in model.layers:
        fully_shard(block, mesh=mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    events = []

    def describe(label, p):
        local = p.to_local() if hasattr(p, "to_local") else p
        events.append(f"{label}: {type(p).__name__} 全局{tuple(p.shape)} 本地{tuple(local.shape)} dtype={p.dtype}")

    def record(name, module):
        describe(name, module.weight)

    for i, block in enumerate(model.layers):
        block.attn.register_forward_pre_hook(lambda m, a, i=i: record(f"层{i}.attn 前向中", m))
        block.ffn[1].register_forward_pre_hook(lambda m, a, i=i: record(f"层{i}.ffn_out 前向中", m))

    if rank == 0:
        describe("前向前常驻", model.layers[0].attn.weight)

    torch.manual_seed(808)
    x = torch.randn(2, 16)
    loss = model(x).sum()
    if rank == 0:
        describe("前向后常驻", model.layers[0].attn.weight)
    loss.backward()
    if rank == 0:
        grad = model.layers[0].attn.weight.grad
        describe("反向后梯度", grad)
    optimizer.step()
    if rank == 0:
        describe("step 后参数", model.layers[0].attn.weight)
        print("  hook / 时点记录：")
        for event in events:
            print(f"    {event}")

    # 每个 rank 都参与 full_tensor 的 all-gather，再在 rank 0 汇总差值。
    final_params = [p.full_tensor().detach() for p in model.parameters()]
    if rank == 0:
        torch.manual_seed(707)
        ref = _Tiny(d=16, n=3)
        torch.manual_seed(808)
        ref_x = torch.randn(2, 16)
        ref_loss = ref(ref_x).sum()
        ref_loss.backward()
        ref_optimizer = torch.optim.AdamW(ref.parameters(), lr=1e-3)
        ref_optimizer.step()
        diffs = [(actual - expected.detach()).abs().max().item()
                 for actual, expected in zip(final_params, ref.parameters())]
        print(f"  loss vs 单进程 fp32 参考: max|diff| = {abs(loss.item() - ref_loss.item()):.2e}（bf16 计算允许舍入差）")
        print(f"  step 后参数 vs 单进程 fp32 参考: max|diff| = {max(diffs):.2e}")


# ---------------------------------------------------------------- driver

def _worker(rank, mode, world, port):
    _init(rank, world, port)
    try:
        if mode == "mesh":
            _mesh(rank)
        elif mode == "dtensor":
            _dtensor(rank)
        elif mode == "meta":
            _meta(rank)
        elif mode == "fsdp":
            _fsdp(rank)
    finally:
        dist.destroy_process_group()


def run(mode):
    world = {"mesh": 8, "dtensor": 2, "meta": 2, "fsdp": 4}[mode]
    mp.spawn(_worker, args=(mode, world, _free_port()), nprocs=world, join=True)


if __name__ == "__main__":
    modes = ["mesh", "dtensor", "meta", "fsdp"]
    parser = argparse.ArgumentParser()
    parser.add_argument("modes", nargs="*", choices=modes, default=None)
    args = parser.parse_args()
    for mode in (args.modes or modes):
        print(f"== {mode} ==")
        run(mode)
