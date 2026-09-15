"""面试手撕代码（15）：手撕 Transformer block 与反向传播 —— NumPy 实现 + torch 对拍。

https://arganzheng.life/coding-interview-transformer-block-and-backprop.html

    python transformer_block.py            # 参数量 / FLOPs 口算 + 数值示例
    python transformer_block.py --check    # 前向与手写反向都与 torch autograd 对拍
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from attention import causal_mask, mha, softmax, split_heads, merge_heads, sdpa


# ---------- 1. 归一化与激活 ----------

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """对最后一维：减均值、除标准差、再仿射。"""
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)                               # 有偏方差（除 N），与 torch 一致
    return (x - mu) / np.sqrt(var + eps) * gamma + beta


def rms_norm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """不减均值、不加 beta：x / sqrt(mean(x²) + eps) * gamma。LLaMA 用它。"""
    rms = np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return x / rms * gamma


def gelu(x: np.ndarray) -> np.ndarray:
    """精确版：x · Φ(x)。tanh 近似见 gelu_tanh。"""
    from math import erf
    return 0.5 * x * (1 + np.vectorize(erf)(x / math.sqrt(2)))


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1 + np.exp(-x))


def swiglu_ffn(x: np.ndarray, w_gate: np.ndarray, w_up: np.ndarray, w_down: np.ndarray) -> np.ndarray:
    """LLaMA 的 FFN：down(silu(x @ gate) * (x @ up))。三个矩阵，中间维通常 8/3·D。"""
    return (silu(x @ w_gate) * (x @ w_up)) @ w_down


# ---------- 2. 一个 GPT block（pre-norm） ----------

def gpt_block(x, p, n_heads):
    """x: (B, T, D)。p 是参数 dict。pre-LN：x + Attn(LN(x))，再 x + FFN(LN(x))。"""
    h = layer_norm(x, p["ln1_g"], p["ln1_b"])
    x = x + mha(h, p["w_qkv"], p["w_o"], n_heads, causal=True)
    h = layer_norm(x, p["ln2_g"], p["ln2_b"])
    x = x + gelu_tanh(h @ p["w_fc"] + p["b_fc"]) @ p["w_proj"] + p["b_proj"]
    return x


def init_block(D: int, rng) -> dict:
    s = 1 / math.sqrt(D)
    return {
        "ln1_g": np.ones(D), "ln1_b": np.zeros(D),
        "w_qkv": rng.standard_normal((D, 3 * D)) * s, "w_o": rng.standard_normal((D, D)) * s,
        "ln2_g": np.ones(D), "ln2_b": np.zeros(D),
        "w_fc": rng.standard_normal((D, 4 * D)) * s, "b_fc": np.zeros(4 * D),
        "w_proj": rng.standard_normal((4 * D, D)) * s / 2, "b_proj": np.zeros(D),
    }


# ---------- 3. 参数量与 FLOPs 口算 ----------

def gpt_params(vocab: int, D: int, L: int, n_ctx: int, tied: bool = True) -> dict:
    """GPT-2 风格（含 bias、learned pos emb、4D FFN）。返回各部分参数量。"""
    attn = 4 * D * D + 4 * D                                     # q k v o 四个 D×D + bias
    ffn = 8 * D * D + 5 * D                                      # D×4D + 4D + 4D×D + D
    ln = 4 * D                                                   # 两个 LN，各 gamma+beta
    per_layer = attn + ffn + ln
    emb = vocab * D + n_ctx * D
    head = 0 if tied else vocab * D
    return {"per_layer": per_layer, "layers": L * per_layer, "embedding": emb, "final_ln": 2 * D,
            "head": head, "total": L * per_layer + emb + 2 * D + head}


def flops_per_token_forward(D: int, L: int, T: int) -> dict:
    """每个 token 的前向 FLOPs（乘加算 2）：线性项 2·12D² 每层 + attention 项 2·2·T·D 每层。"""
    linear = 2 * 12 * D * D * L
    attn = 2 * 2 * T * D * L                                     # QK^T 与 PV 各 2TD
    return {"linear": linear, "attention": attn, "total": linear + attn}


# ---------- 4. 手写反向 ----------

def linear_backward(x, w, dout):
    """y = x @ w。x: (N, in), w: (in, out), dout: (N, out)。"""
    dx = dout @ w.T
    dw = x.T @ dout
    return dx, dw


def softmax_ce_forward_backward(logits, targets):
    """logits: (N, C)，targets: (N,) 类别下标。返回 (loss, dlogits)。dlogits = (p - onehot) / N。"""
    N = logits.shape[0]
    p = softmax(logits)
    loss = -np.log(p[np.arange(N), targets] + 1e-300).mean()
    dlogits = p.copy()
    dlogits[np.arange(N), targets] -= 1
    return loss, dlogits / N


def layer_norm_backward(x, gamma, dout, eps=1e-5):
    """返回 dx, dgamma, dbeta。推导：y = g·x̂ + b，x̂ = (x-μ)/σ；
    dx = (g/σ) · (dŷ - mean(dŷ) - x̂ · mean(dŷ·x̂))，其中 dŷ = dout。"""
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    inv = 1 / np.sqrt(var + eps)
    xhat = (x - mu) * inv
    dgamma = (dout * xhat).reshape(-1, x.shape[-1]).sum(0)
    dbeta = dout.reshape(-1, x.shape[-1]).sum(0)
    dxhat = dout * gamma
    dx = inv * (dxhat - dxhat.mean(-1, keepdims=True) - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, dgamma, dbeta


def two_layer_mlp_train(X, y, hidden, steps, lr, rng):
    """NumPy 手写两层 MLP（ReLU + softmax-CE）训练循环，返回 loss 轨迹。"""
    n_in, n_cls = X.shape[1], int(y.max()) + 1
    W1 = rng.standard_normal((n_in, hidden)) * math.sqrt(2 / n_in)
    b1 = np.zeros(hidden)
    W2 = rng.standard_normal((hidden, n_cls)) * math.sqrt(2 / hidden)
    b2 = np.zeros(n_cls)
    losses = []
    for _ in range(steps):
        z1 = X @ W1 + b1
        a1 = np.maximum(z1, 0)                                   # ReLU
        logits = a1 @ W2 + b2
        loss, dlogits = softmax_ce_forward_backward(logits, y)
        losses.append(loss)
        da1, dW2 = linear_backward(a1, W2, dlogits)
        db2 = dlogits.sum(0)
        dz1 = da1 * (z1 > 0)                                     # ReLU 的导数
        _, dW1 = linear_backward(X, W1, dz1)
        db1 = dz1.sum(0)
        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2
    return losses


# ---------- 5. micrograd 式标量自动求导 ----------

class Value:
    """一个标量 + 它的梯度 + 反向函数。`backward()` 按拓扑序反向传播。"""

    def __init__(self, data, children=(), op=""):
        self.data, self.grad = float(data), 0.0
        self._children, self._op = children, op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward
        return out

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return Value(other) + (-self)
    def __truediv__(self, other): return self * other ** -1
    def __rtruediv__(self, other): return Value(other) * self ** -1

    def backward(self):
        topo, seen = [], set()

        def build(v):
            if v not in seen:
                seen.add(v)
                for c in v._children:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


# ---------- demo & check ----------

def demo():
    print("== GPT-2 small 参数量口算（vocab=50257, D=768, L=12, ctx=1024, tied）")
    p = gpt_params(50257, 768, 12, 1024)
    for k, v in p.items():
        print(f"  {k:10s} {v:>13,}")
    print(f"  每层 ≈ 12·D² = {12 * 768 * 768:,}（bias 与 LN 只占 {(p['per_layer'] - 12 * 768 * 768) / p['per_layer']:.2%}）")

    print("\n== 每 token 前向 FLOPs（D=768, L=12, T=1024）")
    f = flops_per_token_forward(768, 12, 1024)
    print(f"  线性 {f['linear']:,}  attention {f['attention']:,}  合计 {f['total']:,}")
    print(f"  ≈ 2·N_非嵌入 = {2 * p['layers']:,}（线性项）；attention 占 {f['attention'] / f['total']:.1%}")

    print("\n== micrograd：f(a, b) = (a·b + a²).relu() / b，在 a=2, b=3 处的梯度")
    a, b = Value(2.0), Value(3.0)
    f = (a * b + a ** 2).relu() / b
    f.backward()
    print(f"  f = {f.data:.4f}, df/da = {a.grad:.4f}（解析 (b+2a)/b = 7/3）, df/db = {b.grad:.4f}（解析 -a²/b² = -4/9）")

    print("\n== NumPy 两层 MLP 在玩具数据上训练（loss 应单调下降）")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((256, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)                    # XOR 型：线性不可分
    losses = two_layer_mlp_train(X, y, hidden=16, steps=300, lr=0.5, rng=rng)
    print("  loss @ step 0/50/100/200/299:", " ".join(f"{losses[i]:.3f}" for i in (0, 50, 100, 200, 299)))


def check():
    import torch
    import torch.nn.functional as F
    torch.set_default_dtype(torch.float64)

    rng = np.random.default_rng(2)
    B, T, D, H = 2, 5, 16, 4
    x = rng.standard_normal((B, T, D))
    xt = torch.tensor(x, requires_grad=True)

    # LayerNorm / RMSNorm / GELU / SiLU 前向
    g, b = rng.standard_normal(D), rng.standard_normal(D)
    assert np.allclose(layer_norm(x, g, b), F.layer_norm(xt, (D,), torch.tensor(g), torch.tensor(b)).detach().numpy(), atol=1e-10)
    assert np.allclose(rms_norm(x, g), F.rms_norm(xt, (D,), torch.tensor(g), eps=1e-6).detach().numpy(), atol=1e-10)
    assert np.allclose(gelu(x), F.gelu(xt).detach().numpy(), atol=1e-10)
    assert np.allclose(gelu_tanh(x), F.gelu(xt, approximate="tanh").detach().numpy(), atol=1e-10)
    assert np.allclose(silu(x), F.silu(xt).detach().numpy(), atol=1e-10)

    # LayerNorm 反向 vs autograd
    dout = rng.standard_normal((B, T, D))
    gt, bt = torch.tensor(g, requires_grad=True), torch.tensor(b, requires_grad=True)
    y = F.layer_norm(xt, (D,), gt, bt)
    y.backward(torch.tensor(dout))
    dx, dg, db = layer_norm_backward(x, g, dout)
    assert np.allclose(dx, xt.grad.numpy(), atol=1e-10), np.abs(dx - xt.grad.numpy()).max()
    assert np.allclose(dg, gt.grad.numpy(), atol=1e-10) and np.allclose(db, bt.grad.numpy(), atol=1e-10)

    # softmax-CE 反向 vs autograd
    logits = rng.standard_normal((7, 5))
    targets = rng.integers(0, 5, 7)
    lt = torch.tensor(logits, requires_grad=True)
    loss_t = F.cross_entropy(lt, torch.tensor(targets))
    loss_t.backward()
    loss, dlogits = softmax_ce_forward_backward(logits, targets)
    assert abs(loss - loss_t.item()) < 1e-10 and np.allclose(dlogits, lt.grad.numpy(), atol=1e-10)

    # Linear 反向
    w = rng.standard_normal((D, 6))
    xt2 = torch.tensor(x.reshape(-1, D), requires_grad=True)
    wt = torch.tensor(w, requires_grad=True)
    d2 = rng.standard_normal((B * T, 6))
    (xt2 @ wt).backward(torch.tensor(d2))
    dx, dw = linear_backward(x.reshape(-1, D), w, d2)
    assert np.allclose(dx, xt2.grad.numpy()) and np.allclose(dw, wt.grad.numpy())

    # 整个 GPT block 前向 vs 用 torch 算子拼的参考
    p = init_block(D, rng)
    def block_torch(xt):
        P = {k: torch.tensor(v) for k, v in p.items()}
        h = F.layer_norm(xt, (D,), P["ln1_g"], P["ln1_b"])
        qkv = h @ P["w_qkv"]
        q, k, v = qkv.split(D, -1)
        sh = lambda t: t.reshape(B, T, H, D // H).transpose(1, 2)
        a = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), is_causal=True).transpose(1, 2).reshape(B, T, D) @ P["w_o"]
        xt = xt + a
        h = F.layer_norm(xt, (D,), P["ln2_g"], P["ln2_b"])
        return xt + F.gelu(h @ P["w_fc"] + P["b_fc"], approximate="tanh") @ P["w_proj"] + P["b_proj"]
    assert np.allclose(gpt_block(x, p, H), block_torch(torch.tensor(x)).numpy(), atol=1e-10)

    # 参数量：与真的 nn 模块数一数
    D2, H2 = 32, 4
    attn = torch.nn.MultiheadAttention(D2, H2, bias=True)
    ffn = torch.nn.Sequential(torch.nn.Linear(D2, 4 * D2), torch.nn.GELU(), torch.nn.Linear(4 * D2, D2))
    lns = torch.nn.ModuleList([torch.nn.LayerNorm(D2), torch.nn.LayerNorm(D2)])
    n_torch = sum(m.numel() for mod in (attn, ffn, lns) for m in mod.parameters())
    assert n_torch == gpt_params(1, D2, 1, 1)["per_layer"], (n_torch, gpt_params(1, D2, 1, 1)["per_layer"])

    # micrograd vs torch
    a, b = Value(2.0), Value(3.0)
    f = (a * b + a ** 2).relu() / b
    f.backward()
    at, bt = torch.tensor(2.0, requires_grad=True), torch.tensor(3.0, requires_grad=True)
    ft = torch.relu(at * bt + at ** 2) / bt
    ft.backward()
    assert abs(a.grad - at.grad.item()) < 1e-12 and abs(b.grad - bt.grad.item()) < 1e-12
    print("transformer_block.py: all checks passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    check() if args.check else demo()
