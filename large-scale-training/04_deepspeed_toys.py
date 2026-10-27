"""大规模训练工程（04）DeepSpeed — 用 4 个 CPU/gloo rank 演示 ZeRO-3 的分片与参数生命周期。

博客：https://arganzheng.life/deepspeed-architecture-and-source-guide.html

    python 04_deepspeed_toys.py                 # 全部
    python 04_deepspeed_toys.py shards           # zero.Init 风格的逐参数分片
    python 04_deepspeed_toys.py lifecycle        # hook、reduce-scatter、分片 Adam、对照参考
    python 04_deepspeed_toys.py memory           # prefetch 深度 0 / 1 的完整参数驻留字节

Toy 使用 CPU float32，聚焦分片和通信顺序；DeepSpeed 的生产配置使用 bf16 参数。
"""
import argparse
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 4
SHAPES = {"W1": (3, 4), "W2": (4, 2), "b": (2,)}


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _base_parameters():
    gen = torch.Generator().manual_seed(404)
    return {name: torch.randn(shape, generator=gen) * 0.15 for name, shape in SHAPES.items()}


def _local_shard(full, rank):
    flat = full.reshape(-1)
    shard_size = (flat.numel() + WORLD - 1) // WORLD
    padded = torch.zeros(shard_size * WORLD, dtype=flat.dtype)
    padded[:flat.numel()] = flat
    return padded[rank * shard_size:(rank + 1) * shard_size].clone(), flat.numel()


def _gather(shard, numel):
    pieces = [torch.empty_like(shard) for _ in range(WORLD)]
    dist.all_gather(pieces, shard.contiguous())
    return torch.cat(pieces)[:numel]


def _reduce_scatter(grad, shard_size):
    padded = torch.zeros(shard_size * WORLD, dtype=grad.dtype)
    padded[:grad.numel()] = grad.reshape(-1)
    result = torch.empty(shard_size, dtype=grad.dtype)
    reduce_scatter = getattr(dist, "reduce_scatter_single", None) or dist.reduce_scatter_tensor
    reduce_scatter(result, padded)
    return result


class _ParamStore:
    def __init__(self, name, shape, shard, numel):
        self.name = name
        self.shape = shape
        self.shard = shard
        self.numel = numel
        self.data = torch.empty(0)

    def gather(self):
        full = _gather(self.shard, self.numel).view(self.shape)
        self.data = full
        if dist.get_rank() == 0:
            print(f"  {self.name}.data: 空壳 -> 完整 {tuple(full.shape)}（all-gather）")
        return full

    def release(self, event):
        if dist.get_rank() == 0:
            print(f"  {self.name}.data: 完整 -> 空壳（{event}）")
        self.data = torch.empty(0)


class _GatheredLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shard, store):
        weight = store.gather()
        result = x.matmul(weight)
        store.release("前向后 release")
        ctx.store = store
        ctx.save_for_backward(x, shard)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, shard = ctx.saved_tensors
        store = ctx.store
        weight = store.gather()
        grad_x = grad_output.matmul(weight.t())
        grad_weight = x.reshape(-1, x.shape[-1]).t().matmul(
            grad_output.reshape(-1, grad_output.shape[-1]))
        grad_shard = _reduce_scatter(grad_weight, shard.numel())
        store.release("反向 reduce-scatter 后 release")
        return grad_x, grad_shard, None


class _GatheredBias(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shard, store):
        bias = store.gather()
        result = x + bias
        store.release("前向后 release")
        ctx.store = store
        ctx.save_for_backward(shard)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (shard,) = ctx.saved_tensors
        store = ctx.store
        store.gather()
        grad_shard = _reduce_scatter(grad_output.sum(dim=0), shard.numel())
        store.release("反向 reduce-scatter 后 release")
        return grad_output, grad_shard, None


def _init_process(rank, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)


def _shards(rank):
    base = _base_parameters()
    rows = []
    for name, full in base.items():
        shard, numel = _local_shard(full, rank)
        rows.append(f"{name}: param.data={tuple(torch.Size([0]))}, ds_tensor={tuple(shard.shape)}，参数原形状={tuple(full.shape)}，numel={numel}")
    all_rows = [None] * WORLD
    dist.gather_object(rows, all_rows if rank == 0 else None, dst=0)
    if rank == 0:
        print("zero.Init 风格：每个参数保留本 rank 的一段，param.data 是空壳")
        for r, rank_rows in enumerate(all_rows):
            print(f"rank{r}: " + " | ".join(rank_rows))


def _batch(rank, micro):
    gen = torch.Generator().manual_seed(900 + rank * 10 + micro)
    x = torch.randn((2, 3), generator=gen)
    target = torch.randn((2, 2), generator=gen)
    return x, target


def _model(x, stores):
    hidden = torch.relu(_GatheredLinear.apply(x, stores["W1"].shard, stores["W1"]))
    projected = _GatheredLinear.apply(hidden, stores["W2"].shard, stores["W2"])
    return _GatheredBias.apply(projected, stores["b"].shard, stores["b"])


def _reference():
    params = {name: value.clone().requires_grad_(True) for name, value in _base_parameters().items()}
    optimizer = torch.optim.Adam(params.values(), lr=0.01, betas=(0.9, 0.95))
    for rank in range(WORLD):
        for micro in range(2):
            x, target = _batch(rank, micro)
            hidden = torch.relu(x.matmul(params["W1"]))
            prediction = hidden.matmul(params["W2"]) + params["b"]
            ((prediction - target).square().sum() / (WORLD * 2 * 2 * 2)).backward()
    optimizer.step()
    return {name: value.detach() for name, value in params.items()}


def _lifecycle(rank):
    base = _base_parameters()
    stores = {}
    masters = []
    for name, full in base.items():
        shard, numel = _local_shard(full, rank)
        shard.requires_grad_(True)
        stores[name] = _ParamStore(name, tuple(full.shape), shard, numel)
        masters.append(shard)

    if rank == 0:
        print("ZeRO-3 toy：两层 MLP，4 个 rank、每 rank 2 个 micro-batch")
        print("hook 事件（rank0；其余 rank 同步执行同一组 collective）：")
    for micro in range(2):
        x, target = _batch(rank, micro)
        prediction = _model(x, stores)
        ((prediction - target).square().sum() / (WORLD * 2 * 2 * 2)).backward()

    optimizer = torch.optim.Adam(masters, lr=0.01, betas=(0.9, 0.95))
    optimizer.step()
    if rank == 0:
        print("step：Adam 更新各 rank 的 fp32 shard；新值仍留在 ds_tensor，不做全参数 all-gather")
    for store in stores.values():
        store.shard.requires_grad_(False)
        store.shard.copy_(store.shard.detach())

    final = {name: _gather(store.shard.detach(), store.numel).view(store.shape)
             for name, store in stores.items()}
    dist.barrier()
    if rank == 0:
        reference = _reference()
        max_diff = max((final[name] - reference[name]).abs().max().item() for name in final)
        print("末态参数与单进程参考的 max|diff| = %.8f" % max_diff)


def _memory(rank):
    base = _base_parameters()
    sizes = {name: value.numel() * value.element_size() for name, value in base.items()}
    layers = [("layer0", "W1"), ("layer1", "W2+b")]
    group_numel = {"W1": base["W1"].numel(), "W2": base["W2"].numel() + base["b"].numel()}
    group_shards = {}
    for layer, key in layers:
        names = ["W1"] if key == "W1" else ["W2", "b"]
        local = []
        for name in names:
            shard, numel = _local_shard(base[name], rank)
            local.append((name, shard, numel))
        group_shards[layer] = local

    def run(direction, depth):
        live = {}
        events = []
        order = ["layer0", "layer1"] if direction == "forward" else ["layer1", "layer0"]

        def materialize(layer, reason):
            if layer in live:
                return
            full_parts = []
            for name, shard, numel in group_shards[layer]:
                full_parts.append((_gather(shard, numel), name))
            live[layer] = full_parts
            live_bytes = sum(sizes[name] for parts in live.values() for _, name in parts)
            events.append((f"{direction} {reason} {layer}", live_bytes))

        def release(layer):
            live.pop(layer, None)
            live_bytes = sum(sizes[name] for parts in live.values() for _, name in parts)
            events.append((f"{direction} release {layer}", live_bytes))

        for index, layer in enumerate(order):
            materialize(layer, "gather")
            if depth == 1 and index + 1 < len(order):
                materialize(order[index + 1], "prefetch")
            events.append((f"{direction} compute {layer}",
                           sum(sizes[name] for parts in live.values() for _, name in parts)))
            release(layer)
        return events

    outputs = {}
    for depth in (0, 1):
        events = run("forward", depth) + run("backward", depth)
        outputs[depth] = events
    if rank == 0:
        print("每个参数按 float32 计字节；forward / backward 使用同一组 gather / release 规则")
        for depth, events in outputs.items():
            print(f"prefetch_depth={depth}:")
            for event, live_bytes in events:
                print(f"  {event:<32} live_full_parameter_bytes={live_bytes}")
            print(f"  peak={max(value for _, value in events)} bytes")


def _worker(rank, mode, port):
    _init_process(rank, port)
    try:
        {"shards": _shards, "lifecycle": _lifecycle, "memory": _memory}[mode](rank)
    finally:
        dist.destroy_process_group()


def run(mode):
    mp.spawn(_worker, args=(mode, _free_port()), nprocs=WORLD, join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="*", choices=("shards", "lifecycle", "memory"))
    args = parser.parse_args()
    for experiment in args.experiment or ("shards", "lifecycle", "memory"):
        run(experiment)
