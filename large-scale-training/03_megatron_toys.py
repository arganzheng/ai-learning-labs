"""大规模训练工程（03）Megatron-LM — 在一台笔记本上真跑 Megatron 的三件核心机制。

博客：https://arganzheng.life/megatron-lm-architecture-and-source-guide.html

    python 03_megatron_toys.py            # 全部
    python 03_megatron_toys.py groups     # 只跑一个

子实验：
    groups  8 卡 TP2×PP2×DP2，order "tp-cp-ep-dp-pp"：重实现
            generate_masked_orthogonal_rank_groups，打印每个 tp/dp/pp 组的 rank 列表
            （与 Megatron Core 0.18.0 同名函数语义一致，见文件头注释）
    buffer  2 个 gloo 进程（DP=2）：param_data / main_grad 两段扁平 buffer，参数是
            反向顺序的视图，2 个 micro-batch 累加梯度 → reduce-scatter → 每卡只在自己的
            1/2 段上跑 Adam → 拷回 → all-gather，与单进程参考逐格相等
    1f1b    PP=2、m=8 的 1F1B 调度表：warmup / steady / cooldown 三段循环（与
            forward_backward_pipelining_without_interleaving 的结构一致），打印两个
            stage 各自的事件序列

groups 里的 _masked_orthogonal_rank_groups 是 megatron/core/parallel_state.py
generate_masked_orthogonal_rank_groups() 的逐行移植（去掉 unused 参数），
已通过对照 Megatron Core core_v0.18.0 源码手工核对。
"""
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)


def _p(rank, *a):
    if rank == 0:
        print(*a)


# ---------------------------------------------------------------- groups
def _prefix_product(a, init=1):
    r = [init]
    for v in a:
        init = init * v
        r.append(init)
    return r


def _masked_orthogonal_rank_groups(world_size, parallel_size, mask):
    """parallel_state.py 的 generate_masked_orthogonal_rank_groups 逐行移植。

    把 world 看成 shape=parallel_size 的多维网格（order 里靠前的维变化最快）。
    mask=True 的维是「变化的」，其余「固定」；枚举固定维的每一种取值，把变化维
    全部展开，就是一个进程组的 rank 列表。
    """
    def decompose(index, shape, stride=None):
        if stride is None:
            stride = _prefix_product(shape)
        idx = [(index // d) % s for s, d in zip(shape, stride)]
        return idx

    masked_shape = [s for s, m in zip(parallel_size, mask) if m]
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]
    global_stride = _prefix_product(parallel_size)
    masked_stride = [d for d, m in zip(global_stride, mask) if m]
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]

    group_size = 1
    for s in masked_shape:
        group_size *= s
    groups = []
    for group_index in range(world_size // group_size):
        coord = decompose(group_index, unmasked_shape)
        base = sum(x * y for x, y in zip(coord, unmasked_stride))
        group = []
        for rank_in_group in range(group_size):
            coord_in = decompose(rank_in_group, masked_shape)
            rank = base + sum(x * y for x, y in zip(coord_in, masked_stride))
            group.append(rank)
        groups.append(group)
    return groups


def groups():
    """8 卡 TP2 × DP2 × PP2，order 'tp-cp-ep-dp-pp'（CP=EP=1 退化掉）。"""
    # order tp-cp-ep-dp-pp 的 parallel_size = [tp, cp, ep, dp, pp]
    tp, cp, ep, dp, pp = 2, 1, 1, 2, 2
    sizes = [tp, cp, ep, dp, pp]
    world = tp * cp * ep * dp * pp
    print(f"world_size={world}, parallel_size(tp,cp,ep,dp,pp)={sizes}, order='tp-cp-ep-dp-pp'")
    print("网格坐标：rank = tp_r + dp_r*2 + pp_r*4  (CP=EP=1 退化)")
    print("  rank0..7 的 (tp,dp,pp)：",
          [(r % 2, (r // 2) % 2, r // 4) for r in range(world)])
    for name, mask in [("tp", [True, False, False, False, False]),
                       ("dp", [False, False, False, True, False]),
                       ("pp", [False, False, False, False, True]),
                       ("dp-cp", [False, True, False, True, False])]:
        g = _masked_orthogonal_rank_groups(world, sizes, mask)
        print(f"{name} 组（{len(g)} 个，各 {len(g[0])} 卡）: {g}")


# ---------------------------------------------------------------- buffer
def _buffer_worker(rank, port):
    """DP=2：两段扁平 buffer + reduce-scatter + 分片 Adam + all-gather。"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    torch.manual_seed(0)

    _rs = getattr(dist, "reduce_scatter_single", None) or dist.reduce_scatter_tensor
    _ag = getattr(dist, "all_gather_single", None) or dist.all_gather_into_tensor

    # 三个「参数」：W1 (3x2) W2 (2x2) WL (2x6)。numel=6+4+12=22（已是 dp 倍数）
    # 按反向顺序排进 buffer：WL 在前（后层先反向）。分片边界=11 落在 WL 中间：
    # WL 元素 0..10 归 rank0、元素 11 归 rank1——分片边界不对齐参数边界。
    shapes = {"W1": (3, 2), "W2": (2, 2), "WL": (2, 6)}
    params = {k: torch.randn(*s, dtype=torch.bfloat16) for k, s in shapes.items()}
    # 参考值（单进程，rank0 打印时对照）
    ref_params = {k: v.clone().float() for k, v in params.items()}
    torch.manual_seed(1)
    inputs = [torch.randn(4, 3), torch.randn(4, 3)]  # 两个 micro-batch
    targets = [torch.randn(4, 6), torch.randn(4, 6)]

    def fwd(b, p):
        # y = relu(relu(x @ W1) @ W2) @ WL，只关心梯度流：每个参数都吃到梯度
        h = torch.relu(b @ p["W1"].float())
        z = torch.relu(h @ p["W2"].float())
        return z @ p["WL"].float()

    # ---- 参考：单进程两个 micro-batch 累加、Adam 一步 ----
    if rank == 0:
        ref = {k: v.clone().requires_grad_(True) for k, v in ref_params.items()}
        opt = torch.optim.Adam(ref.values(), lr=0.1, betas=(0.9, 0.95))
        for b, t in zip(inputs, targets):
            loss = (fwd(b, ref) - t).pow(2).sum()
            loss.backward()
        opt.step()
        for k in ref:
            print(f"[ref] {k}.grad sum = {ref[k].grad.float().sum().item():.4f}  "
                  f"{k}.data sum = {ref[k].data.sum().item():.4f}")
        globals()["_ref"] = ref

    # ---- 并行版：参数是 buffer 的视图，按反向顺序 ----
    numel = sum(p.numel() for p in params.values())  # 22
    pad = (-numel) % 2                               # pad 到 dp=2 的倍数 -> 24
    param_data = torch.zeros(numel + pad, dtype=torch.bfloat16)
    grad_data = torch.zeros(numel + pad, dtype=torch.float32)
    order = ["WL", "W2", "W1"]                       # 反向顺序：后层在前
    views, off = {}, 0
    for k in order:
        n = params[k].numel()
        param_data[off:off + n] = params[k].flatten()
        views[k] = param_data[off:off + n].view(shapes[k])
        views[k].main_grad = grad_data[off:off + n].view(shapes[k])
        globals()["_off_" + k] = off
        off += n
    half = (numel + pad) // 2
    _p(rank, f"numel={numel} pad={pad} param_data/grad_data 各 {numel + pad} 个元素")
    _p(rank, "buffer 内偏移（反向顺序，元素下标）:", {k: (globals()['_off_' + k], globals()['_off_' + k] + views[k].numel()) for k in order})
    wl_n = views['WL'].numel()
    _p(rank, f"分片边界={half}：WL(0..{wl_n - 1}) 跨片——元素 0..{half - 1} 归 rank0、{half}..{wl_n - 1} 归 rank1"
             if half < wl_n else f"分片边界={half}：WL 完全在 rank0（0..{wl_n - 1}）")

    # 两个 micro-batch 的前向+反向：梯度直接写进 main_grad 视图（梯度累积融合的等价物）
    for i, (b, t) in enumerate(zip(inputs, targets)):
        vp = {k: v.clone().float().requires_grad_(True) for k, v in views.items()}
        loss = (fwd(b, vp) - t).pow(2).sum()
        loss.backward()
        for k in order:
            views[k].main_grad += vp[k].grad
        _p(rank, f"micro-batch {i}: loss={loss.item():.4f}，各参数 dW 已累加进 main_grad")

    # reduce-scatter grad_data：每卡拿到自己 1/2 段的归约和
    shard = torch.zeros(half, dtype=torch.float32)
    _rs(shard, grad_data)
    _p(rank, f"rank{rank} 拿到 grad_data[{rank * half}:{rank * half + half}]（reduce-scatter）")

    # 每卡在自己那段上：建 fp32 主参数分片 + Adam + 更新
    seg = slice(rank * half, rank * half + half)
    main = param_data[seg].float().clone().requires_grad_(True)
    opt = torch.optim.Adam([main], lr=0.1, betas=(0.9, 0.95))
    main.grad = shard.clone()
    opt.step()
    param_data[seg] = main.detach().to(torch.bfloat16)
    _p(rank, f"rank{rank} Adam 更新了 fp32 主参数分片（{shard.numel()} 个元素）")

    # all-gather：各 rank 的 1/2 段拼回完整 param_data
    _ag(param_data, param_data[seg].clone())
    _p(rank, f"rank{rank} all-gather 后 param_data sum={param_data.float().sum().item():.4f}")

    # 对照参考
    dist.barrier()
    if rank == 0:
        ref = globals()["_ref"]
        ok = True
        for k in order:
            got = views[k].float()
            diff = (got - ref[k].data).abs().max().item()
            ok &= diff < 2e-2
            print(f"  {k}: 并行 vs 单进程 max|diff| = {diff:.5f}")
        print("两步 Adam + reduce-scatter/all-gather 与单进程一致:", ok)


def buffer():
    mp.spawn(_buffer_worker, args=(19317,), nprocs=2, join=True)


# ---------------------------------------------------------------- 1f1b
def schedule_1f1b(pp_size, num_microbatches):
    """forward_backward_pipelining_without_interleaving 的三段循环，翻译成事件序列。"""
    seqs = {}
    for stage in range(pp_size):
        num_warmup = min(pp_size - stage - 1, num_microbatches)
        remaining = num_microbatches - num_warmup
        ev = []
        fid, bid = 0, 0
        for _ in range(num_warmup):
            ev.append(("warmup", f"F{fid}")); fid += 1
        for _ in range(remaining):
            ev.append(("steady", f"F{fid}")); fid += 1
            ev.append(("steady", f"B{bid}")); bid += 1
        for _ in range(num_warmup):
            ev.append(("cooldown", f"B{bid}")); bid += 1
        seqs[stage] = ev
    return seqs


def f1b():
    seqs = schedule_1f1b(2, 8)
    for stage, ev in seqs.items():
        warm = [e for ph, e in ev if ph == "warmup"]
        steady = [e for ph, e in ev if ph == "steady"]
        cool = [e for ph, e in ev if ph == "cooldown"]
        print(f"stage{stage}: warmup {warm} | steady {steady} | cooldown {cool}")
        print(f"          序列: {' '.join(e for _, e in ev)}")
    print("warmup 数 = pp_size - stage - 1：stage0 预热 1 个前向，stage1 不预热直接交替")


if __name__ == "__main__":
    all_exp = {"groups": groups, "buffer": buffer, "1f1b": f1b}
    sel = sys.argv[1:] or list(all_exp)
    for name in sel:
        print(f"===== {name} =====")
        all_exp[name]()
        print()
