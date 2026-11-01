"""大规模训练工程（01）显存账与 MFU — 在一台笔记本上把四笔账各量一次。

博客：https://arganzheng.life/training-state-anatomy-memory-and-mfu.html

    python 01_state_toys.py             # 全部
    python 01_state_toys.py bytes       # 只跑一个

子实验：
    bytes       小型 Llama 风格模型（L=2, h=256, f=688, V=1000）：bf16 计算参数 +
                fp32 主参数 + AdamW(m,v) 按 Megatron 式混合精度手动管理，跑一个
                step 后遍历全部张量统计真实字节，与 2N/4N/12N 的公式逐项对照
    activations Korthikanti et al. 2022 论文口径的 GPT 块（LayerNorm、多头注意力
                含 dropout、4h GELU MLP 含 dropout，bf16）：用 saved_tensors_hooks
                统计一层为反向实际保存的张量字节，按来源分组，与 sbh(34 + 5as/h)
                逐项对照——只如实测量，不凑公式
    flops       torch.utils.flop_counter 统计小模型前向+反向的真实 FLOP，
                与 6N·tokens + 注意力项对照
    timeline    一个 step 内「已分配字节」的时间线：saved_tensors_hooks 记激活
                的进出，post-accumulate hook 记梯度到账，输出曲线给 step-timeline 图
"""
import sys
from contextlib import contextmanager

import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=140)

# 小模型尺寸：能看清每一笔账，又小到瞬间跑完
L, H, A, KV, FDIM, V = 2, 256, 4, 2, 688, 1000


def gb(n):
    return n / 1e9


def _block(h=H, f=FDIM, a=A, kv=KV):
    """一个 Llama 风格的 Transformer 层：RMSNorm、GQA 注意力、SwiGLU MLP。"""
    import torch.nn as nn

    d = h // a
    kv_dim = kv * d
    return nn.ModuleDict(dict(
        ln1=nn.LayerNorm(h), q=nn.Linear(h, h, bias=False),
        k=nn.Linear(h, kv_dim, bias=False), v=nn.Linear(h, kv_dim, bias=False),
        o=nn.Linear(h, h, bias=False),
        ln2=nn.LayerNorm(h), gate=nn.Linear(h, f, bias=False),
        up=nn.Linear(h, f, bias=False), down=nn.Linear(f, h, bias=False),
    ))


def _forward_block(blk, x, s_heads=A):
    """朴素注意力（物化分数矩阵）+ SwiGLU MLP，按子模块逐步写，便于挂钩子。"""
    b, s, h = x.shape
    d = h // s_heads
    kv_h = blk["k"].out_features // d

    def attn(t):
        t = blk["ln1"](t)
        q = blk["q"](t).view(b, s, s_heads, d).transpose(1, 2)
        k = blk["k"](t).view(b, s, kv_h, d).transpose(1, 2)
        v = blk["v"](t).view(b, s, kv_h, d).transpose(1, 2)
        k = k.repeat_interleave(s_heads // kv_h, dim=1)
        v = v.repeat_interleave(s_heads // kv_h, dim=1)
        sc = torch.softmax(q @ k.transpose(-1, -2) / d ** 0.5, dim=-1)
        t = blk["o"]((sc @ v).transpose(1, 2).reshape(b, s, h))
        return x + t

    def mlp(t):
        t = blk["ln2"](t)
        return t + blk["down"](F.silu(blk["gate"](t)) * blk["up"](t))

    return mlp(attn(x))


def toy_bytes():
    """bf16 参数 + bf16 梯度 + fp32 主参数 + fp32 m/v：每参数 16 字节。"""
    print(f"== bytes：小模型 L={L} h={H} f={FDIM} V={V}，Megatron 式混合精度手动管理")
    blocks = torch.nn.ModuleList([_block() for _ in range(L)]).bfloat16()
    emb = torch.nn.Embedding(V, H).bfloat16()
    head = torch.nn.Linear(H, V, bias=False).bfloat16()
    params = [p for mod in (blocks, emb, head) for p in mod.parameters()]
    n = sum(p.numel() for p in params)
    # fp32 主参数与 Adam 两个矩：优化器持有，常驻
    master = [p.detach().clone().float() for p in params]
    m = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    v = [torch.zeros_like(p, dtype=torch.float32) for p in params]

    x = torch.randint(0, V, (1, 64))
    h_ = emb(x)
    for blk in blocks:
        h_ = _forward_block(blk, h_)
    loss = head(h_).sum()
    loss.backward()
    # AdamW 在 fp32 主参数上更新，结果 cast 回 bf16 计算参数
    for p, w, mm, vv in zip(params, master, m, v):
        g = p.grad.float()
        mm.mul_(0.9).add_(g, alpha=0.1)
        vv.mul_(0.999).addcmul_(g, g, value=0.001)
        w.mul_(0.99).add_(mm, alpha=-1e-3)
        p.data.copy_(w.bfloat16())

    def nbytes(ts):
        return sum(t.numel() * t.element_size() for t in ts)

    grads = [p.grad for p in params]
    rows = [("bf16 参数", params, 2), ("bf16 梯度", grads, 2),
            ("fp32 主参数", master, 4), ("Adam m", m, 4), ("Adam v", v, 4)]
    total = 0
    for name, ts, coef in rows:
        real = nbytes(ts)
        total += real
        print(f"  {name:　<10} {real / 1e6:8.2f} MB   公式 {coef}N = {coef * n / 1e6:8.2f} MB   {'一致' if real == coef * n else '不一致'}")
    print(f"  常驻合计        {total / 1e6:8.2f} MB   16N = {16 * n / 1e6:.2f} MB   N = {n / 1e6:.2f} M")
    print(f"  若梯度 fp32（Megatron 默认 main_grad）：18N = {18 * n / 1e6:.2f} MB（多出 2N = {2 * n / 1e6:.2f} MB）")
    print()


# ----------------------------------------------------------------- activations
_REGION = [None]
_SAVED = []          # (region, [bytes...])


def _pack(t):
    _SAVED.append((_REGION[0], t.numel() * t.element_size(), isinstance(t, torch.nn.Parameter), tuple(t.shape), str(t.dtype)))
    return t


def _unpack(t):
    return t


@contextmanager
def _region(name):
    prev = _REGION[0]
    _REGION[0] = name
    try:
        yield
    finally:
        _REGION[0] = prev


def toy_activations():
    """一层 GPT 块为反向实际保存的字节，按来源分组，对照 sbh(34 + 5as/h)。"""
    s, b, h, a, f = 64, 2, 256, 4, 4 * H
    print(f"== activations：一层 GPT 块 s={s} b={b} h={h} a={a} f={f}（朴素注意力、有 dropout，bf16）")
    w = lambda o, i=h: torch.nn.Parameter(torch.randn(i, o, dtype=torch.bfloat16))
    wq, wk, wv, wo = w(h), w(h), w(h), w(h)
    w1, w2 = w(f), w(h, f)
    ln1w, ln1b = torch.nn.Parameter(torch.ones(h, dtype=torch.bfloat16)), torch.nn.Parameter(torch.zeros(h, dtype=torch.bfloat16))
    ln2w, ln2b = torch.nn.Parameter(torch.ones(h, dtype=torch.bfloat16)), torch.nn.Parameter(torch.zeros(h, dtype=torch.bfloat16))
    x = torch.randn(b, s, h, dtype=torch.bfloat16, requires_grad=True)
    d = h // a

    from torch.autograd.graph import saved_tensors_hooks
    _SAVED.clear()
    with saved_tensors_hooks(_pack, _unpack):
        with _region("ln1"):
            t1 = F.layer_norm(x, (h,), ln1w, ln1b)
        with _region("qkv"):
            q = t1 @ wq
            k = t1 @ wk
            v = t1 @ wv
        q = q.view(b, s, a, d).transpose(1, 2)
        k = k.view(b, s, a, d).transpose(1, 2)
        v = v.view(b, s, a, d).transpose(1, 2)
        with _region("qk^t"):
            sc = q @ k.transpose(-1, -2) / d ** 0.5
        with _region("softmax"):
            p = torch.softmax(sc, dim=-1)
        with _region("attn_dropout"):
            p = F.dropout(p, 0.1)
        with _region("pv"):
            o = p @ v
        with _region("out_proj"):
            o = o.transpose(1, 2).reshape(b, s, h) @ wo
        with _region("out_dropout"):
            o = F.dropout(o, 0.1)
        x2 = x + o
        with _region("ln2"):
            t2 = F.layer_norm(x2, (h,), ln2w, ln2b)
        with _region("fc1"):
            g = t2 @ w1
        with _region("gelu"):
            g = F.gelu(g)
        with _region("fc2"):
            out = g @ w2
        with _region("mlp_dropout"):
            out = F.dropout(out, 0.1)
        y = x2 + out
        y.sum().backward()

    # 分组打印；权重是参数不是激活，单列
    order = ["ln1", "qkv", "qk^t", "softmax", "attn_dropout", "pv", "out_proj",
             "out_dropout", "ln2", "fc1", "gelu", "fc2", "mlp_dropout"]
    groups = {r: [] for r in order}
    for r, nb, is_p, shape, dtype in _SAVED:
        groups[r].append((nb, is_p, shape, dtype))
    formula = {  # Korthikanti 表里的每 token 字节（×sb 得总量）
        "ln1": 2 * h, "qkv": 2 * h, "qk^t": 4 * h, "softmax": 2 * a * s,
        "attn_dropout": a * s, "pv": 2 * h + 2 * a * s, "out_proj": 2 * h,
        "out_dropout": 1 * h, "ln2": 2 * h, "fc1": 2 * h, "gelu": 8 * h,
        "fc2": 8 * h, "mlp_dropout": 1 * h,
    }
    note = {
        "qkv": "三个投影各自保存相同输入；论文按共享输入计一次",
        "attn_dropout": "公式按 bool mask 1 B/元素；CPU 的 _dropout_impl 按输入 dtype 存 bf16（2 B）",
        "pv": "V + dropout 后的注意力矩阵（公式分别计 2h + 2as）",
        "out_proj": "线性层为 wgrad 保存输入",
        "ln1": "除输入 2h 外还保存 mean/rstd（实测 bf16，各 1 个/位置）",
        "ln2": "同上",
        "out_dropout": "同上（CUDA 的 fused native_dropout 才存 bool 1 B）",
        "mlp_dropout": "同上",
    }
    tot, tot_p, tot_f = 0, 0, 0
    for r in order:
        real = sum(nb for nb, is_p, _, _ in groups[r] if not is_p)
        wbytes = sum(nb for nb, is_p, _, _ in groups[r] if is_p)
        detail = ", ".join(f"{shape}:{dtype}" for _, is_p, shape, dtype in groups[r] if not is_p)
        tot += real
        tot_p += wbytes
        exp = formula[r] * s * b
        tot_f += exp
        print(f"  {r:　<14} 保存 {len(groups[r])} 个  激活 {real / 1e3:9.1f} KB   公式项 {formula[r]:>5} B/tok → {exp / 1e3:8.1f} KB   {note.get(r, '')}")
        if detail:
            print(f"    张量：{detail}")
    print(f"  激活小计：实测 {tot / 1e3:.1f} KB / 层，公式 sbh(34 + 5as/h) = {tot_f / 1e3:.1f} KB，比值 {tot / tot_f:.3f}")
    print(f"  另保存权重（是参数不是激活，不计入公式）：{tot_p / 1e3:.1f} KB —— autograd 为 dgrad/wgrad 也要留 W")
    print()


# ----------------------------------------------------------------- flops
def toy_flops():
    """FlopCounterMode 实测前向+反向 FLOP，与 6N·tokens + 注意力项对照。"""
    from torch.utils.flop_counter import FlopCounterMode
    s, b = 64, 2
    print(f"== flops：小模型 L={L} h={H} f={FDIM} V={V}，s={s} b={b}")
    blocks = torch.nn.ModuleList([_block() for _ in range(L)])
    emb = torch.nn.Embedding(V, H)
    head = torch.nn.Linear(H, V, bias=False)
    x = torch.randint(0, V, (b, s))
    n = sum(p.numel() for mod in (blocks, emb, head) for p in mod.parameters())
    # 参与 GEMM 的参数 = 非 embedding 参数 + 输出层（与正文 §六.1 同口径）
    n_gemm = sum(p.numel() for mod in (blocks, head) for p in mod.parameters())
    tokens = s * b

    def run():
        h_ = emb(x)
        for blk in blocks:
            h_ = _forward_block(blk, h_)
        return head(h_).sum()

    fc = FlopCounterMode(display=False)
    with fc:
        loss = run()
    fwd = fc.get_total_flops()
    fc2 = FlopCounterMode(display=False)
    with fc2:
        loss.backward()
    bwd = fc2.get_total_flops()
    print(f"  实测 FLOP：前向 {fwd / 1e9:.3f} GFLOP，反向 {bwd / 1e9:.3f} GFLOP，合计 {(fwd + bwd) / 1e9:.3f}")
    print(f"  6N·tokens（N 全部参数）:          {6 * n * tokens / 1e9:.3f} GFLOP")
    ref = 6 * n_gemm * tokens + 12 * L * s * H * tokens
    print(f"  6·N_gemm·tokens + 注意力(全 mask)： {ref / 1e9:.3f} GFLOP")
    print(f"  实测 / 6N = {(fwd + bwd) / (6 * n * tokens):.3f}   实测 / (6N+注意力) = {(fwd + bwd) / ref:.3f}")
    print()


# ----------------------------------------------------------------- timeline
def timeline_curve():
    """一个 step 内活跃张量的曲线，由 saved-tensor 与 grad hooks 记录。

    返回 [(事件序号, 阶段, 激活字节, 梯度字节)] 以及参数量；常驻项按 14N
    （bf16 参数 2N + fp32 主参数 4N + Adam m/v 8N）计算，画图时另行缩放。
    """
    n_layer = 8
    blocks = torch.nn.ModuleList([_block() for _ in range(n_layer)]).bfloat16()
    x = torch.randn(1, 32, H, dtype=torch.bfloat16)
    params = [p for blk in blocks for p in blk.parameters()]
    n = sum(p.numel() for p in params)
    master = [p.detach().float().clone() for p in params]
    moments = [(torch.zeros_like(p, dtype=torch.float32), torch.zeros_like(p, dtype=torch.float32))
               for p in params]
    param_ptrs = {p.untyped_storage().data_ptr() for p in params}
    resident = 2 * n + 4 * n + 8 * n
    act_live, grads_live = [0], [0]
    live_storages = {}
    log = [(0, "开始", 0, 0)]

    def pack(t):
        ptr = t.untyped_storage().data_ptr()
        if ptr not in param_ptrs:
            size = t.untyped_storage().nbytes()
            if ptr not in live_storages:
                live_storages[ptr] = [size, 0]
                act_live[0] += size
            live_storages[ptr][1] += 1
        return t

    def unpack(t):
        ptr = t.untyped_storage().data_ptr()
        if ptr in live_storages:
            live_storages[ptr][1] -= 1
            if live_storages[ptr][1] == 0:
                act_live[0] -= live_storages.pop(ptr)[0]
        return t

    def mark(phase):
        log.append((len(log), phase, act_live[0], grads_live[0]))

    for p in params:
        def on_grad(param):
            grads_live[0] += param.grad.untyped_storage().nbytes()
            mark("反向")
        p.register_post_accumulate_grad_hook(on_grad)

    from torch.autograd.graph import saved_tensors_hooks
    with saved_tensors_hooks(pack, unpack):
        for blk in blocks:
            x = _forward_block(blk, x)
            mark("前向")
        x.sum().backward()

    with torch.no_grad():
        for p, w, (m, v) in zip(params, master, moments):
            m.mul_(0.9).add_(p.grad.float(), alpha=0.1)
            v.mul_(0.999).addcmul_(p.grad.float(), p.grad.float(), value=0.001)
            w.mul_(0.99).add_(m, alpha=-1e-3)
            p.copy_(w.bfloat16())
            p.grad = None
    grads_live[0] = 0
    mark("step 后")
    return log, n, resident


def toy_timeline():
    log, n, resident = timeline_curve()
    print(f"== timeline：L=8 小模型一个 step 的实测激活 / 梯度（参数 N={n / 1e6:.2f} M；常驻参数+主参数+m/v={resident / 1e6:.2f} MB）")
    for i, phase, act, gr in log:
        print(f"  {i:3d} {phase:6s} 激活 {act / 1e6:8.3f} MB  梯度 {gr / 1e6:8.3f} MB  张量合计 {(resident + act + gr) / 1e6:8.3f} MB")
    print()


ALL = {"bytes": toy_bytes, "activations": toy_activations, "flops": toy_flops, "timeline": toy_timeline}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in ALL.items():
        if which in ("all", name):
            fn()
