"""面试手撕代码（17）：手撕损失函数与训练算法 —— NumPy 实现 + torch 对拍。

https://arganzheng.life/coding-interview-losses-and-training-algorithms.html

    python losses_training.py            # 数值示例
    python losses_training.py --check    # 与 torch / trl 公式对拍
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from attention import softmax, log_softmax


# ---------- 1. 分类损失 ----------

def cross_entropy(logits: np.ndarray, targets: np.ndarray, label_smoothing: float = 0.0, ignore_index: int = -100) -> float:
    """logits (N, C)，targets (N,)。平滑：目标分布 = (1-ε)·onehot + ε/C。ignore_index 的样本不计入。"""
    mask = targets != ignore_index
    lp = log_softmax(logits[mask])
    t = targets[mask]
    nll = -lp[np.arange(len(t)), t]
    if label_smoothing == 0:
        return float(nll.mean())
    smooth = -lp.mean(-1)                                        # 对均匀分布的交叉熵
    return float(((1 - label_smoothing) * nll + label_smoothing * smooth).mean())


def kl_div(log_p: np.ndarray, log_q: np.ndarray) -> float:
    """KL(p || q) = Σ p (log p - log q)，对 batch 取平均。输入都是 log 概率 (N, C)。"""
    p = np.exp(log_p)
    return float((p * (log_p - log_q)).sum(-1).mean())


def binary_cross_entropy_with_logits(z: np.ndarray, y: np.ndarray) -> float:
    """数值稳定：max(z,0) - z·y + log(1 + exp(-|z|))。"""
    return float((np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))).mean())


def focal_loss(logits: np.ndarray, targets: np.ndarray, gamma: float = 2.0) -> float:
    """-(1-p_t)^γ log p_t：难样本（p_t 小）权重大。"""
    lp = log_softmax(logits)[np.arange(len(targets)), targets]
    p = np.exp(lp)
    return float((-(1 - p) ** gamma * lp).mean())


# ---------- 2. 对比学习 ----------

def info_nce(q: np.ndarray, k: np.ndarray, temperature: float = 0.07) -> float:
    """q, k: (N, d) 已归一化。第 i 个 q 的正样本是第 i 个 k，其余 N-1 个是负样本。= CE(q k^T / τ, arange(N))。"""
    logits = q @ k.T / temperature
    return cross_entropy(logits, np.arange(len(q)))


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


# ---------- 3. 偏好与 RL ----------

def dpo_loss(logp_chosen: np.ndarray, logp_rejected: np.ndarray, ref_logp_chosen: np.ndarray, ref_logp_rejected: np.ndarray, beta: float = 0.1) -> float:
    """-log σ(β [(log π(y_w) - log π_ref(y_w)) - (log π(y_l) - log π_ref(y_l))])。输入都是序列 log 概率之和 (N,)。"""
    margin = beta * ((logp_chosen - ref_logp_chosen) - (logp_rejected - ref_logp_rejected))
    return float(-np.log(1 / (1 + np.exp(-margin))).mean())     # -log sigmoid(margin)


def gae(rewards: np.ndarray, values: np.ndarray, gamma: float = 0.99, lam: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """一条轨迹。values 长 T+1（最后一个是 bootstrap）。δ_t = r_t + γ V_{t+1} - V_t；A_t = δ_t + γλ A_{t+1}。返回 (advantages, returns)。"""
    T = len(rewards)
    adv = np.zeros(T)
    last = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values[:-1]


def ppo_clip_loss(logp_new: np.ndarray, logp_old: np.ndarray, adv: np.ndarray, eps: float = 0.2) -> float:
    """-mean(min(r·A, clip(r, 1-ε, 1+ε)·A))，r = exp(logp_new - logp_old)。"""
    ratio = np.exp(logp_new - logp_old)
    return float(-np.minimum(ratio * adv, np.clip(ratio, 1 - eps, 1 + eps) * adv).mean())


def grpo_advantages(rewards: np.ndarray) -> np.ndarray:
    """同一 prompt 的 G 个回答的奖励 (G,)：组内标准化 (r - mean) / (std + ε)。没有 critic。"""
    return (rewards - rewards.mean()) / (rewards.std() + 1e-8)


# ---------- 4. 优化器与调度 ----------

class AdamW:
    """一步：m = β1 m + (1-β1) g；v = β2 v + (1-β2) g²；偏差修正；θ -= lr (m̂ / (√v̂ + ε) + wd θ)。"""

    def __init__(self, params: dict, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.p, self.lr, self.b1, self.b2, self.eps, self.wd = params, lr, betas[0], betas[1], eps, weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict):
        self.t += 1
        for k in self.p:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.wd * self.p[k])   # 解耦的 weight decay


def cosine_with_warmup(step: int, warmup: int, total: int, lr_max: float, lr_min: float = 0.0) -> float:
    if step < warmup:
        return lr_max * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * min(1.0, progress)))


def clip_grad_norm(grads: dict, max_norm: float) -> float:
    """全局范数裁剪：所有梯度拼成一个向量的 L2 范数超过 max_norm 就整体缩放。返回裁剪前的范数。"""
    total = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for k in grads:
            grads[k] *= scale
    return total


# ---------- 5. LoRA ----------

class LoRALinear:
    """y = x W + (α/r) · x A B。W 冻结；A (in, r) 高斯初始化，B (r, out) 零初始化 → 初始时 ΔW = 0。"""

    def __init__(self, w: np.ndarray, r: int, alpha: float, rng):
        self.w = w
        self.a = rng.standard_normal((w.shape[0], r)) / math.sqrt(r)
        self.b = np.zeros((r, w.shape[1]))
        self.scale = alpha / r

    def forward(self, x):
        return x @ self.w + self.scale * (x @ self.a) @ self.b

    def merge(self) -> np.ndarray:
        """推理时合并：W' = W + (α/r) A B，零额外开销。"""
        return self.w + self.scale * self.a @ self.b


# ---------- demo & check ----------

def demo():
    rng = np.random.default_rng(0)
    print("== 交叉熵与 label smoothing（N=4, C=5）")
    logits = rng.standard_normal((4, 5)) * 2
    targets = np.array([0, 3, 1, 4])
    for eps in (0.0, 0.1):
        print(f"  ε={eps}: CE = {cross_entropy(logits, targets, eps):.4f}")
    print(f"  KL(p || p) = {kl_div(log_softmax(logits), log_softmax(logits)):.1e}；KL(p || uniform) = {kl_div(log_softmax(logits), np.full_like(logits, -math.log(5))):.4f}")

    print("\n== InfoNCE：正样本相似度越高、损失越低（N=8, d=16, τ=0.07）")
    q = l2_normalize(rng.standard_normal((8, 16)))
    for noise in (1.0, 0.3, 0.05):
        k = l2_normalize(q + noise * rng.standard_normal(q.shape))
        print(f"  noise={noise}: loss = {info_nce(q, k):.4f}（随机基线 ln 8 = {math.log(8):.4f}）")

    print("\n== DPO：chosen 相对 ref 提升越多、损失越低（β=0.1）")
    ref_c, ref_r = np.array([-50.0]), np.array([-52.0])
    for dc, dr in ((0.0, 0.0), (5.0, -5.0), (20.0, -20.0)):
        print(f"  Δchosen={dc:+.0f}, Δrejected={dr:+.0f}: loss = {dpo_loss(ref_c + dc, ref_r + dr, ref_c, ref_r):.4f}")

    print("\n== GAE（γ=0.99, λ=0.95）在一条 5 步轨迹上")
    r = np.array([0.0, 0.0, 0.0, 0.0, 1.0]); v = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.0])
    adv, ret = gae(r, v)
    print("  δ 由后往前累积，adv =", np.round(adv, 4).tolist())
    print("  returns = adv + V     =", np.round(ret, 4).tolist())

    print("\n== PPO clip：ratio 超出 [0.8, 1.2] 后梯度被截断")
    adv = np.array([1.0]); old = np.array([0.0])
    for new in (0.0, 0.1, 0.3, 0.5):
        print(f"  ratio={math.exp(new):.3f}: loss = {ppo_clip_loss(np.array([new]), old, adv):.4f}")

    print("\n== GRPO 组内优势：rewards [1, 0, 0, 1, 0] →", np.round(grpo_advantages(np.array([1.0, 0, 0, 1, 0])), 4).tolist())

    print("\n== AdamW 在 f(x) = (x-3)² 上 200 步（lr=0.1）")
    p = {"x": np.array([0.0])}
    opt = AdamW(p, lr=0.1, weight_decay=0.0)
    for _ in range(200):
        opt.step({"x": 2 * (p["x"] - 3)})
    print(f"  x = {p['x'][0]:.4f}（最优 3）")

    print("\n== cosine + warmup（warmup 3, total 10, lr_max 1）")
    print("  ", [f"{cosine_with_warmup(s, 3, 10, 1.0):.3f}" for s in range(10)])

    print("\n== LoRA：初始 ΔW = 0，训练 B 后 merge 与不 merge 结果一致")
    w = rng.standard_normal((8, 6)); x = rng.standard_normal((3, 8))
    lora = LoRALinear(w, r=2, alpha=4, rng=rng)
    print(f"  初始 |y_lora - y_base| = {np.abs(lora.forward(x) - x @ w).max():.1e}")
    lora.b = rng.standard_normal(lora.b.shape)
    print(f"  merge 后 |x W' - forward| = {np.abs(x @ lora.merge() - lora.forward(x)).max():.1e}；额外参数 {lora.a.size + lora.b.size} vs 全量 {w.size}")


def check():
    import torch
    import torch.nn.functional as F
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(4)

    logits = rng.standard_normal((6, 7)); targets = rng.integers(0, 7, 6); targets[2] = -100
    lt, tt = torch.tensor(logits), torch.tensor(targets)
    for eps in (0.0, 0.1):
        assert abs(cross_entropy(logits, targets, eps) - F.cross_entropy(lt, tt, label_smoothing=eps, ignore_index=-100).item()) < 1e-10
    lp, lq = log_softmax(logits), log_softmax(rng.standard_normal((6, 7)))
    assert abs(kl_div(lp, lq) - F.kl_div(torch.tensor(lq), torch.tensor(lp), log_target=True, reduction="batchmean").item()) < 1e-10
    z, y = rng.standard_normal(9) * 5, rng.integers(0, 2, 9).astype(float)
    assert abs(binary_cross_entropy_with_logits(z, y) - F.binary_cross_entropy_with_logits(torch.tensor(z), torch.tensor(y)).item()) < 1e-10

    # InfoNCE == CE over similarity matrix
    q = l2_normalize(rng.standard_normal((5, 8))); k = l2_normalize(rng.standard_normal((5, 8)))
    assert abs(info_nce(q, k, 0.1) - F.cross_entropy(torch.tensor(q @ k.T / 0.1), torch.arange(5)).item()) < 1e-10

    # DPO == -logsigmoid(beta * logits)（trl 的 sigmoid loss）
    a, b, c, d = (rng.standard_normal(4) for _ in range(4))
    ref = -F.logsigmoid(0.1 * ((torch.tensor(a) - torch.tensor(c)) - (torch.tensor(b) - torch.tensor(d)))).mean().item()
    assert abs(dpo_loss(a, b, c, d, 0.1) - ref) < 1e-10

    # GAE：与显式 Σ (γλ)^l δ_{t+l} 对拍
    r = rng.standard_normal(6); v = rng.standard_normal(7)
    adv, _ = gae(r, v, 0.9, 0.8)
    delta = r + 0.9 * v[1:] - v[:-1]
    explicit = np.array([sum((0.9 * 0.8) ** l * delta[t + l] for l in range(6 - t)) for t in range(6)])
    assert np.allclose(adv, explicit)

    # PPO：ratio 在区间内时等于 -ratio·A；越界时梯度为 0（数值检查）
    adv = np.array([1.0, -1.0])
    assert abs(ppo_clip_loss(np.array([0.05, 0.05]), np.zeros(2), adv) - (-(math.exp(0.05) * 1 + math.exp(0.05) * -1) / 2)) < 1e-12
    assert abs(ppo_clip_loss(np.array([0.5]), np.zeros(1), np.array([1.0])) - (-1.2)) < 1e-12        # A>0 越上界：clip 到 1.2

    # GRPO：均值 0、方差 1
    g = grpo_advantages(rng.standard_normal(8))
    assert abs(g.mean()) < 1e-10 and abs(g.std() - 1) < 1e-6

    # AdamW vs torch.optim.AdamW（同一初值、同一梯度序列）
    w0 = rng.standard_normal(5)
    mine = {"w": w0.copy()}
    opt = AdamW(mine, lr=0.01, weight_decay=0.1)
    wt = torch.tensor(w0.copy(), requires_grad=True)
    opt_t = torch.optim.AdamW([wt], lr=0.01, weight_decay=0.1)
    for _ in range(10):
        g = rng.standard_normal(5)
        opt.step({"w": g})
        wt.grad = torch.tensor(g); opt_t.step(); opt_t.zero_grad()
    assert np.allclose(mine["w"], wt.detach().numpy(), atol=1e-12), np.abs(mine["w"] - wt.detach().numpy()).max()

    # 梯度裁剪 vs clip_grad_norm_
    grads = {"a": rng.standard_normal((3, 3)) * 5, "b": rng.standard_normal(4) * 5}
    ta, tb = torch.tensor(grads["a"].copy(), requires_grad=True), torch.tensor(grads["b"].copy(), requires_grad=True)
    ta.grad, tb.grad = ta.detach().clone(), tb.detach().clone()
    total_t = torch.nn.utils.clip_grad_norm_([ta, tb], 1.0).item()
    total = clip_grad_norm(grads, 1.0)
    assert abs(total - total_t) < 1e-10 and np.allclose(grads["a"], ta.grad.numpy(), atol=1e-8) and np.allclose(grads["b"], tb.grad.numpy(), atol=1e-8)

    # cosine schedule vs 手算
    assert abs(cosine_with_warmup(3, 3, 13, 1.0) - 1.0) < 1e-12 and abs(cosine_with_warmup(8, 3, 13, 1.0) - 0.5) < 1e-12 and abs(cosine_with_warmup(13, 3, 13, 1.0)) < 1e-12

    # LoRA merge
    w = rng.standard_normal((8, 6)); x = rng.standard_normal((3, 8))
    lora = LoRALinear(w, 2, 4, rng); lora.b = rng.standard_normal(lora.b.shape)
    assert np.allclose(x @ lora.merge(), lora.forward(x))
    print("losses_training.py: all checks passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    check() if args.check else demo()
