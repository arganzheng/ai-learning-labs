"""大规模训练工程（02）并行策略全景 — 在一台笔记本上用 4 个 CPU 进程真跑五种并行。

博客：https://arganzheng.life/parallelism-strategies-which-state-to-shard.html

没有 GPU 也能跑：torch.distributed 的 gloo 后端在 CPU 进程之间做的 all-reduce /
reduce-scatter / all-gather / all-to-all / send-recv 与 NCCL 在 GPU 之间做的语义完全一样，
只是慢。每个子实验都把「并行版」的结果与「单进程算一遍」的参考值对数，误差打印出来。

    python 02_parallelism_toys.py            # 全部
    python 02_parallelism_toys.py tp ep      # 只跑几个

子实验：
    zero     DP / ZeRO-1/2/3 放置、逐原语通信字节与参数校验
    tp       列切 + 行切的 MLP：f/g 共轭与单卡逐格相等
    ring/cp  Ring Attention：K/V 沿环 send/recv，online-softmax 合并
    ulysses  按序列切 → all-to-all → 按头注意力 → all-to-all 回来
    bubble   GPipe / 1F1B / interleaved 气泡率与真实 PyTorch 1F1B schedule
    mesh     Megatron rank 顺序下 2 节点 × 4 卡的 TP/DP/PP 组
    pp       GPipe：4 个 stage、4 个 micro-batch，激活前传、梯度回传
    ep       专家并行：token all-to-all 去专家所在进程再回来
"""
import math
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 4
torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)


def _init(rank, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    torch.manual_seed(0)  # 每个进程同一份种子 → 各进程能各自造出同一份「完整模型」当参考


# PyTorch 2.14 改了两个原语的名字，旧名仍可用但会警告；两者都兼容
_rs = getattr(dist, "reduce_scatter_single", None) or dist.reduce_scatter_tensor
_ag = getattr(dist, "all_gather_single", None) or dist.all_gather_into_tensor


def _p(rank, *a):
    if rank == 0:
        print(*a)


# ---------------------------------------------------------------- zero
def zero(rank, port):
    """同一个 8 参数小模型在 DP / ZeRO-1 / ZeRO-2 / ZeRO-3 下各跑一步。

    打印每 rank 常驻的参数/梯度/优化器元素数与通信字节（AR / RS / AG 分开计），
    验证四种放置更新后的参数与单进程一致、通信量为 2N/2N/2N/3N。
    """
    _init(rank, port)
    # ---- 先保留原来的小实验：all-reduce = reduce-scatter + all-gather
    g = torch.arange(8, dtype=torch.float32) + 10 * rank  # 进程 r 的梯度：10r, 10r+1, …, 10r+7
    ar = g.clone()
    dist.all_reduce(ar)  # ① DP：每个进程拿到 8 个元素的和
    shard = torch.zeros(2)
    _rs(shard, g)  # ② ZeRO：进程 r 只拿到第 r 段（2 个元素）的和
    my_update = shard * 0.5  # ③ 只更新自己那 1/4 的参数（这里用 ×0.5 代替优化器）
    full = torch.zeros(8)
    _ag(full, my_update)  # ④ 把 4 段更新后的参数拼回完整的一份
    _p(rank, "-- RS + AG == AR：")
    _p(rank, "all-reduce 结果        :", ar.tolist())
    _p(rank, "reduce-scatter 段(进程0):", shard.tolist(), "← 正是 all-reduce 结果的前两个元素")
    _p(rank, "all-gather 后 0.5×和   :", full.tolist())
    _p(rank, "RS + AG == AR × 0.5 ?  :", torch.allclose(full, ar * 0.5))

    # ---- 同一个小模型：8 个 fp32 参数、梯度 8 个、优化器状态 m+v 16 个
    N = 8
    w0 = torch.arange(N, dtype=torch.float32) + 1.0   # 完整参数（各进程同一份）
    def step(mode):
        """在 4 个 rank 上模拟 mode 下的一步，返回 (更新后参数, 通信字节计数 dict)。"""
        comm = {"all_reduce": 0, "reduce_scatter": 0, "all_gather": 0}
        if mode == "DP":
            p = w0.clone()                       # 每卡完整参数 8 + 完整梯度 8 + 完整优化器 16
            local = p * (rank + 1) + rank               # 各卡不同的梯度（数据不同）
            gg = local.clone(); dist.all_reduce(gg); comm["all_reduce"] += 2 * (WORLD - 1) / WORLD * N
            upd = p - 0.1 * gg                   # 每卡各自更新全部参数
            held = (N, N, 2 * N)                 # (参数, 梯度, 优化器) 常驻元素数
        elif mode == "ZeRO-1":
            p = w0.clone()
            local = p * (rank + 1) + rank
            seg = torch.zeros(N // WORLD)
            _rs(seg, local); comm["reduce_scatter"] += (WORLD - 1) / WORLD * N
            own = w0[rank * N // WORLD : (rank + 1) * N // WORLD] - 0.1 * seg
            newp = torch.zeros(N)
            _ag(newp, own); comm["all_gather"] += (WORLD - 1) / WORLD * N
            upd = newp
            held = (N, N, 2 * N // WORLD)
        elif mode == "ZeRO-2":
            p = w0.clone()
            local = p * (rank + 1) + rank
            seg = torch.zeros(N // WORLD)
            _rs(seg, local); comm["reduce_scatter"] += (WORLD - 1) / WORLD * N
            own = w0[rank * N // WORLD : (rank + 1) * N // WORLD] - 0.1 * seg
            newp = torch.zeros(N)
            _ag(newp, own); comm["all_gather"] += (WORLD - 1) / WORLD * N
            upd = newp
            held = (N, N // WORLD, 2 * N // WORLD)
        else:  # ZeRO-3
            ps = w0[rank * N // WORLD : (rank + 1) * N // WORLD].clone()   # 常驻 1/4 参数
            full_p = torch.zeros(N)
            _ag(full_p, ps); comm["all_gather"] += (WORLD - 1) / WORLD * N  # 前向前拼参数
            full_p2 = torch.zeros(N)
            _ag(full_p2, ps); comm["all_gather"] += (WORLD - 1) / WORLD * N  # 反向前再拼
            local = full_p2 * (rank + 1) + rank
            seg = torch.zeros(N // WORLD)
            _rs(seg, local); comm["reduce_scatter"] += (WORLD - 1) / WORLD * N
            upd_shard = ps - 0.1 * seg
            upd = torch.zeros(N)
            # 更新完不需要 all-gather：下步前向再拼；为对比参数，这里 gather 一次收输出（不计通信）
            _ag(upd, upd_shard)
            held = (N // WORLD, N // WORLD, 2 * N // WORLD)
        return upd, comm, held

    ref = w0 - 0.1 * (10.0 * w0 + 6.0)  # 归约梯度 = w0·(1+2+3+4) + (0+1+2+3)
    rows = []
    for mode in ("DP", "ZeRO-1", "ZeRO-2", "ZeRO-3"):
        upd, comm, held = step(mode)
        err = (upd - ref).abs().max().item()
        buf = [None] * WORLD
        dist.all_gather_object(buf, (mode, held, comm, err))
        rows.append(buf[0])
    if rank == 0:
        unit_bytes = (WORLD - 1) / WORLD * N * 4  # 此 toy 用 fp32；单个 ≈N 集合通信的精确字节数
        print("-- 同一小模型（8 参数，fp32）一步；常驻列为每 rank 元素数，通信列为每卡发出字节：")
        print(f"{'模式':6s} {'参数':>4s} {'梯度':>4s} {'优化器':>5s} {'AR B':>6s} {'RS B':>6s} {'AG B':>6s} {'≈xN':>5s}  参数与单进程最大误差")
        for mode, held, comm, err in rows:
            total_bytes = sum(comm.values()) * 4
            print(f"{mode:6s} {held[0]:4d} {held[1]:4d} {held[2]:5d} {comm['all_reduce']*4:6.0f} {comm['reduce_scatter']*4:6.0f} {comm['all_gather']*4:6.0f} {total_bytes/unit_bytes:5.1f}  {err:.1e}")
        print("（≈xN 用 4 卡精确值 (n−1)/n·N = 6 个 fp32 元素折算；每卡通信字节按每个原语分别计）")
    dist.destroy_process_group()


# ---------------------------------------------------------------- tp
class _F(torch.autograd.Function):
    """Megatron mappings.py 的 _CopyToModelParallelRegion：前向恒等，反向 all-reduce。"""

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, gx):
        gx = gx.clone()
        dist.all_reduce(gx)
        return gx


class _G(torch.autograd.Function):
    """_ReduceFromModelParallelRegion：前向 all-reduce，反向恒等。"""

    @staticmethod
    def forward(ctx, x):
        x = x.clone()
        dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, gx):
        return gx


def tp(rank, port):
    """X(1×2) → A(2×2) 列切 → ReLU → B(2×2) 行切 → Z(1×2)，用 2 个进程；另 2 个进程旁观。"""
    _init(rank, port)
    X = torch.tensor([[1.0, 2.0]])
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 完整权重（每个进程都造得出，用来当参考）
    B = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    # 参考：单卡
    Xr, Ar, Br = X.clone().requires_grad_(), A.clone().requires_grad_(), B.clone().requires_grad_()
    Zr = torch.relu(Xr @ Ar) @ Br
    Zr.sum().backward()
    # TP：只用进程 0、1（N_t = 2）；进程 2、3 只参加集合通信、贡献 0
    group = dist.new_group([0, 1])
    t = rank if rank < 2 else None
    if t is not None:
        Ai = A[:, t : t + 1].clone().requires_grad_()  # ① 列切：进程 t 持 A 的第 t 列（2×1）
        Bi = B[t : t + 1, :].clone().requires_grad_()  # ② 行切：进程 t 持 B 的第 t 行（1×2）
        Xi = X.clone().requires_grad_()

        class F(_F):
            @staticmethod
            def backward(ctx, gx):
                gx = gx.clone()
                dist.all_reduce(gx, group=group)
                return gx

        class G(_G):
            @staticmethod
            def forward(ctx, x):
                x = x.clone()
                dist.all_reduce(x, group=group)
                return x

        Yi = torch.relu(F.apply(Xi) @ Ai)  # ③ f：前向恒等；Y_i = X A_i 是 Y 的第 t 列，无通信
        Zi = Yi @ Bi  # ④ 行切：部分和 Y_i B_i
        Z = G.apply(Zi)  # ⑤ g：前向 all-reduce，两份部分和相加得完整 Z
        Z.sum().backward()  # 反向：g 恒等 → dB_i、dY_i 本地 → f 对 dX 做 all-reduce
        out = dict(Yi=Yi.detach(), Zi=Zi.detach(), Z=Z.detach(), dAi=Ai.grad, dBi=Bi.grad, dX=Xi.grad)
    else:
        out = None
    gathered = [None] * WORLD
    dist.all_gather_object(gathered, out)
    if rank == 0:
        o0, o1 = gathered[0], gathered[1]
        print("X =", X.tolist(), " A =", A.tolist(), " B =", B.tolist())
        print("前向  进程0: Y_0 =", o0["Yi"].tolist(), " Z_0 = Y_0 B_0 =", o0["Zi"].tolist())
        print("      进程1: Y_1 =", o1["Yi"].tolist(), " Z_1 = Y_1 B_1 =", o1["Zi"].tolist())
        print("      g all-reduce → Z =", o0["Z"].tolist(), " 单卡 Z =", Zr.detach().tolist())
        print("反向  dB_0 =", o0["dBi"].tolist(), " dB_1 =", o1["dBi"].tolist(), " 单卡 dB =", Br.grad.tolist(), "（按行拼起来）")
        print("      dA_0 =", o0["dAi"].tolist(), " dA_1 =", o1["dAi"].tolist(), " 单卡 dA =", Ar.grad.tolist(), "（按列拼起来）")
        print("      f all-reduce → dX =", o0["dX"].tolist(), " 单卡 dX =", Xr.grad.tolist())
        ok = (
            torch.allclose(o0["Z"], Zr)
            and torch.allclose(torch.cat([o0["dAi"], o1["dAi"]], 1), Ar.grad)
            and torch.allclose(torch.cat([o0["dBi"], o1["dBi"]], 0), Br.grad)
            and torch.allclose(o0["dX"], Xr.grad)
        )
        print("TP 前向 + 反向与单卡逐格相等:", ok)
    dist.destroy_process_group()


# ---------------------------------------------------------------- cp
def _merge(m1, l1, acc1, m2, l2, acc2):
    """online softmax：把两块 (最大值 m, 分母 l, 加权和 acc) 合成一块。"""
    m = torch.maximum(m1, m2)
    c1, c2 = torch.exp(m1 - m), torch.exp(m2 - m)
    return m, l1 * c1 + l2 * c2, acc1 * c1[:, None] + acc2 * c2[:, None]


def _block_attn(q, k, v):
    s = q @ k.T  # 分数 [nq, nk]（省略 1/√d，小例子里用不着）
    m = s.max(1).values
    p = torch.exp(s - m[:, None])
    return m, p.sum(1), p @ v


def cp(rank, port):
    """4 个进程各持 2 个 token 的 Q/K/V；K/V 沿环传 3 步；与完整注意力对数。"""
    _init(rank, port)
    s, d = 8, 4
    Q, K, V = torch.randn(s, d), torch.randn(s, d), torch.randn(s, d)  # 同种子 → 每进程同一份完整序列
    ref = torch.softmax(Q @ K.T, 1) @ V  # 单卡完整注意力
    n = s // WORLD
    q = Q[rank * n : (rank + 1) * n]  # ① 本卡只持自己那段 Q
    k, v = K[rank * n : (rank + 1) * n].clone(), V[rank * n : (rank + 1) * n].clone()
    m, l, acc = _block_attn(q, k, v)  # ② 先用本地 K/V 块算一块
    nxt, prv = (rank + 1) % WORLD, (rank - 1) % WORLD
    log = [rank]
    for step in range(WORLD - 1):
        kv = torch.cat([k, v], 1)
        buf = torch.empty_like(kv)
        req = dist.isend(kv, nxt)  # ③ 把手上的 K/V 块发给下一张卡……
        dist.recv(buf, prv)  # ……同时接收上一张卡的（真实实现里这一步与 ② 的计算重叠）
        req.wait()
        k, v = buf[:, :d], buf[:, d:]
        src = (rank - step - 1) % WORLD
        log.append(src)
        m, l, acc = _merge(m, l, acc, *_block_attn(q, k, v))  # ④ 用 online softmax 把新块合进来
    out = acc / l[:, None]
    err = (out - ref[rank * n : (rank + 1) * n]).abs().max()
    logs = [None] * WORLD
    dist.all_gather_object(logs, (log, err.item()))
    if rank == 0:
        print("序列 8 个 token，4 张卡各 2 个；每步每卡用的 K/V 块（第一个数是本卡的）：")
        for lg, _ in logs:
            print(f"  卡 {lg[0]}: K/V 块 {lg}")
        print("Ring 4 步合并后与完整注意力的最大误差（各卡）:", [f"{e:.1e}" for _, e in logs])
    dist.destroy_process_group()


# ---------------------------------------------------------------- pp
def pp(rank, port):
    """4 个 stage 各一层 Linear(4,4)，m = 4 个 micro-batch，GPipe：先全部前向再全部反向。"""
    _init(rank, port)
    h, m, b = 4, 4, 2
    Ws = [torch.randn(h, h) * 0.5 for _ in range(WORLD)]  # 完整模型的 4 层（每进程都造得出）
    xs = [torch.randn(b, h) for _ in range(m)]  # 4 个 micro-batch
    # 参考：单卡
    Wr = [w.clone().requires_grad_() for w in Ws]
    loss = 0
    for x in xs:
        for w in Wr:
            x = torch.tanh(x @ w)
        loss = loss + x.sum()
    loss.backward()
    # PP：进程 r 只持第 r 层
    W = Ws[rank].clone().requires_grad_()
    ins, outs = [], []
    for i in range(m):  # ---- 前向：m 个 micro-batch 依次流过
        if rank == 0:
            x = xs[i]
        else:
            x = torch.empty(b, h)
            dist.recv(x, rank - 1)  # ① 收上一 stage 的激活
            x.requires_grad_()
        y = torch.tanh(x @ W)
        if rank < WORLD - 1:
            dist.send(y, rank + 1)  # ② 发给下一 stage
        ins.append(x)
        outs.append(y)
    for i in range(m):  # ---- 反向：GPipe 先做完全部前向才开始
        if rank == WORLD - 1:
            gy = torch.ones(b, h)  # loss = sum → dL/dy = 1
        else:
            gy = torch.empty(b, h)
            dist.recv(gy, rank + 1)  # ③ 收下一 stage 传回的激活梯度
        outs[i].backward(gy)  # ④ 本 stage 反向，梯度累积进 W.grad
        if rank > 0:
            dist.send(ins[i].grad, rank - 1)  # ⑤ 把对输入的梯度传给上一 stage
    err = (W.grad - Wr[rank].grad).abs().max().item()
    errs = [None] * WORLD
    dist.all_gather_object(errs, err)
    if rank == 0:
        print("4 stage × 4 micro-batch，GPipe 调度；各 stage 权重梯度与单卡的最大误差:", [f"{e:.1e}" for e in errs])
        print("每个 stage 只保存了自己那一层的参数与 4 个 micro-batch 的输入激活（反向要用）")
    dist.destroy_process_group()


# ---------------------------------------------------------------- ep
class _A2A(torch.autograd.Function):
    """all-to-all（按 split 长度）；反向是 split 互换的 all-to-all——把梯度按原路送回。"""

    @staticmethod
    def forward(ctx, x, out_splits, in_splits):
        ctx.splits = (in_splits, out_splits)
        out = x.new_empty(sum(out_splits), x.shape[1])
        dist.all_to_all_single(out, x.contiguous(), out_splits, in_splits)
        return out

    @staticmethod
    def backward(ctx, g):
        in_splits, out_splits = ctx.splits
        gx = g.new_empty(sum(in_splits), g.shape[1])
        dist.all_to_all_single(gx, g.contiguous(), in_splits, out_splits)
        return gx, None, None


def ep(rank, port):
    """E = 8 个专家、k = 2，4 个进程每个持 2 个专家、各有 16 个 token。"""
    _init(rank, port)
    E, k, T, h = 8, 2, 16, 4
    Wexp = torch.randn(E, h, h) * 0.5  # 完整专家表（每进程都造得出，当参考）
    Wr = Wexp.clone().requires_grad_()
    torch.manual_seed(100 + rank)
    x = torch.randn(T, h)  # 本进程的 token
    Wrouter = torch.randn(h, E) if rank == 0 else None
    Wrouter = [Wrouter]
    dist.broadcast_object_list(Wrouter, 0)
    logits = x @ Wrouter[0]
    prob = torch.softmax(logits, 1)
    w, eid = prob.topk(k, 1)  # ① 路由：每个 token 选 2 个专家与权重
    # 参考：每个 token 自己去查所有专家（等价于 EP 组内只有一张卡）
    ref = torch.zeros(T, h)
    for t in range(T):
        for j in range(k):
            ref[t] += w[t, j] * torch.tanh(x[t] @ Wr[eid[t, j]])
    ref.sum().backward()  # 本进程 token 对全部专家的梯度
    ref_grad = Wr.grad.clone()
    dist.all_reduce(ref_grad)  # 四个进程的 token 加起来，才是每个专家的完整梯度
    # EP：进程 r 持专家 2r、2r+1
    per = E // WORLD
    Wl = Wexp[rank * per : (rank + 1) * per].clone().requires_grad_()
    flat_e = eid.reshape(-1)  # ② 每个 token 复制 k 份，一份去一个专家
    flat_w = w.reshape(-1)
    flat_x = x.repeat_interleave(k, 0)
    dst = flat_e // per  # 目标进程 = 专家编号 // 每进程专家数
    order = torch.argsort(dst, stable=True)  # ③ 按目标进程排好序，才能一次 all-to-all
    send_splits = torch.bincount(dst, minlength=WORLD).tolist()
    recv_t = torch.zeros(WORLD, dtype=torch.long)
    dist.all_to_all_single(recv_t, torch.tensor(send_splits))  # 先交换「我要给你几个」，才知道收多少
    recv_splits = recv_t.tolist()
    sent_e = flat_e[order]
    recv_e = torch.zeros(sum(recv_splits), dtype=torch.long)
    dist.all_to_all_single(recv_e, sent_e, recv_splits, send_splits)  # 专家编号也一起送过去
    xin = _A2A.apply(flat_x[order], recv_splits, send_splits)  # ④ dispatch：token 去专家所在进程
    yout = torch.zeros_like(xin)
    for j in range(per):  # ⑤ 本地专家逐个算自己的一批 token（真实实现是 grouped GEMM）
        sel = recv_e == rank * per + j
        yout[sel] = torch.tanh(xin[sel] @ Wl[j])
    yback = _A2A.apply(yout, send_splits, recv_splits)  # ⑥ combine：结果按原路送回
    y = torch.zeros(T * k, h)
    y[order] = yback  # 还原发送前的顺序
    out = (flat_w[:, None] * y).reshape(T, k, h).sum(1)  # ⑦ 按路由权重加权求和
    out.sum().backward()
    err_f = (out - ref).abs().max().item()
    err_b = (Wl.grad - ref_grad[rank * per : (rank + 1) * per]).abs().max().item()
    info = [None] * WORLD
    dist.all_gather_object(info, (send_splits, recv_splits, err_f, err_b, torch.bincount(flat_e, minlength=E).tolist()))
    if rank == 0:
        print(f"E = {E} 个专家、k = {k}，4 个进程各持 {per} 个专家、各 {T} 个 token（每 token 复制 {k} 份 → 每进程发出 {T*k}）")
        print("dispatch 矩阵：第 i 行 = 进程 i 发给进程 0..3 的 token 数")
        for i, (s, _, _, _, _) in enumerate(info):
            print(f"  进程 {i}: {s}   合计 {sum(s)}")
        recv = [sum(r) for _, r, _, _, _ in info]
        avg = T * k * WORLD / WORLD
        print(f"各进程收到的 token 数: {recv}   平均 {avg:.0f}，最忙 / 平均 = ρ = {max(recv)/avg:.2f}")
        load = torch.tensor([c for *_, c in info]).sum(0).tolist()
        cf = 1.25
        C = math.ceil(cf * T * WORLD * k / E)
        print(f"每个专家收到的 token 数: {load}   平均 {T*WORLD*k/E:.0f}；容量因子 {cf} → 容量 C = ⌈{cf}×{T*WORLD}×{k}/{E}⌉ = {C}，"
              f"超过 C 的专家: {[e for e, c in enumerate(load) if c > C]}")
        print("EP 前向与「每个 token 自己算」的最大误差（各进程）:", [f"{e:.1e}" for *_, e, _, _ in info])
        print("EP 本地专家梯度与完整梯度的最大误差（各进程）  :", [f"{e:.1e}" for *_, e, _ in info])
    dist.destroy_process_group()


# ---------------------------------------------------------------- ulysses
def ulysses(rank, port):
    """4 进程各持 s/4 个 token 的全部头；all-to-all 换成「全部 token 的 1 个头」算注意力再换回。"""
    _init(rank, port)
    s, a, d = 8, 4, 4   # 8 token、4 头、每头 dim=4
    Q, K, V = torch.randn(s, a, d), torch.randn(s, a, d), torch.randn(s, a, d)
    # 参考：单进程逐头完整注意力 → [s, a, d]
    ref = torch.stack([torch.softmax(Q[:, i] @ K[:, i].T, -1) @ V[:, i] for i in range(a)], 1)
    n = s // WORLD

    def to_heads(x):
        """[n, a, d]（按序列切，本卡一段 token 的全部头）→ [s, d]（按头切，本卡一个头的全部 token）。

        gloo all_to_all_single 按第 0 维均分：发送 buffer 第 j 段去进程 j。
        发送：第 j 段 = 本卡第 j 头的 n 个 token → x.transpose(0,1) 的头主序。
        接收：第 j 段 = 进程 j 的第 rank 头（它的 seq 块）→ 拼成全部 s 个 token 的第 rank 头。"""
        send = x.transpose(0, 1).contiguous().reshape(-1)   # 头主序：段 j = 头 j
        buf = torch.empty_like(send)
        dist.all_to_all_single(buf, send)
        return buf.reshape(s, d)                           # 全部 token、本卡第 rank 头

    qs = Q[rank * n:(rank + 1) * n]
    ks = K[rank * n:(rank + 1) * n]
    vs = V[rank * n:(rank + 1) * n]
    qh, kh, vh = to_heads(qs), to_heads(ks), to_heads(vs)  # 各 [8, 4]
    oh = torch.softmax(qh @ kh.T, -1) @ vh                 # 本卡那一个头的完整注意力 [8, 4]

    def to_seq(o):
        """[s, d]（按头切）→ [n, a, d]（按序列切）。

        发送：第 j 段 = 进程 j 的 token 块的本卡头输出 → o 的序列主序第 j 块。
        接收：第 j 段 = 进程 j 的第 j 头、本卡 token 块 → reshape 后按头转置回来。"""
        send = o.contiguous().reshape(-1)                  # 序列主序：段 j = 第 j 块
        buf = torch.empty_like(send)
        dist.all_to_all_single(buf, send)
        return buf.reshape(WORLD, n, d).transpose(0, 1)    # [头, token, d] → [token, 头, d]

    out = to_seq(oh)
    err = (out - ref[rank * n:(rank + 1) * n]).abs().max().item()
    info = [None] * WORLD
    dist.all_gather_object(info, (tuple(qs.shape), tuple(qh.shape), tuple(out.shape), err))
    if rank == 0:
        s0, s1, s2 = info[0][:3]
        print(f"按序列切 {list(s0)} → all-to-all → 按头切 [{s1[0]}, 1, {s1[1]}]（本卡 1 个头、全部 token）→ all-to-all 回 {list(s2)}")
        print("Ulysses 与单进程完整注意力的最大误差（各卡）:", [f"{e:.1e}" for *_, e in info])
    dist.destroy_process_group()


# ---------------------------------------------------------------- bubble
def bubble(rank, port):
    """打印 GPipe / 1F1B / interleaved 的气泡率表与 p=4、m=4 的 1F1B 逐拍时间表。"""
    _init(rank, port)
    if rank == 0:
        print("气泡率（占总时间，t_b = t_f）：(p-1)/(v·m + p - 1)")
        print(f"{'p':>3s} {'m':>4s} {'v':>3s} {'GPipe/1F1B':>10s} {'interleaved':>11s}")
        for p, m, v in [(4, 4, 1), (4, 8, 1), (8, 8, 1), (8, 16, 1), (16, 16, 1),
                        (16, 16, 8), (16, 135, 1), (16, 32, 8)]:
            b1 = (p - 1) / (m + p - 1)
            bv = (p - 1) / v / (m + (p - 1) / v) if v > 1 else b1
            print(f"{p:3d} {m:4d} {v:3d} {b1:10.1%} {bv:11.1%}")
        print("\nPyTorch 真生成：Schedule1F1B._get_pipeline_order(p=4, m=4)（_format_pipeline_order 输出）：")
        from types import SimpleNamespace
        from torch.distributed.pipelining.schedules import Schedule1F1B, _format_pipeline_order
        sched = Schedule1F1B(SimpleNamespace(num_stages=4), n_microbatches=4)
        print(_format_pipeline_order(sched.pipeline_order))
    dist.destroy_process_group()


# ---------------------------------------------------------------- mesh
def _decompose(idx, shape):
    out = []
    for s in reversed(shape):
        out.append(idx % s)
        idx //= s
    return out[::-1]


def _rank_groups(world_size, sizes):
    """parallel_state.py 的 generate_masked_orthogonal_rank_groups（无 mask 退化情形）。

    order 'tp-cp-ep-dp-pp'：sizes 靠前的维变化最快。返回各维的 rank 组。"""
    assert math.prod(sizes) == world_size
    groups = {}
    for i, s in enumerate(sizes):
        if s == 1:
            continue
        stride = math.prod(sizes[:i])
        rest = world_size // (stride * s)
        g = []
        for outer in range(world_size // (stride * s)):
            for inner in range(stride):
                base = outer * stride * s + inner
                g.append([base + j * stride for j in range(s)])
        groups[i] = g
    return groups


def mesh(rank, port):
    """2 节点 × 4 卡的五元组分组：tp=2, cp=1, ep=1, dp=2, pp=2，order 'tp-cp-ep-dp-pp'。"""
    _init(rank, port)
    if rank == 0:
        world = 8
        sizes = [2, 1, 1, 2, 2]  # tp, cp, ep, dp, pp
        print(f"world={world}（2 节点 × 4 卡），parallel_size(tp,cp,ep,dp,pp)={sizes}，order='tp-cp-ep-dp-pp'")
        print("rank = tp_r + 2·dp_r + 4·pp_r；TP 最内、TP rank 相邻。物理节点按 rank // 4 映射（连续 rank 在同一节点）。")
        names = {0: "TP", 3: "DP", 4: "PP"}
        groups = _rank_groups(world, sizes)
        node_of = {r: r // 4 for r in range(world)}
        for i, gname in names.items():
            for g in groups.get(i, []):
                node = "同节点" if len({node_of[r] for r in g}) == 1 else "跨节点"
                print(f"  {gname} 组 {g}  {node}")
        for node in range(2):
            print(f"节点 {node} ranks: {[r for r in range(world) if node_of[r] == node]}")
    dist.destroy_process_group()


# ---------------------------------------------------------------- main
EXPS = {"zero": zero, "tp": tp, "cp": cp, "ring": cp, "pp": pp, "ep": ep,
        "ulysses": ulysses, "bubble": bubble, "mesh": mesh}

if __name__ == "__main__":
    names = sys.argv[1:] or list(EXPS)
    for i, name in enumerate(names):
        print(f"\n===== {name} =====")
        mp.spawn(EXPS[name], args=(29600 + i,), nprocs=WORLD)
