"""RoPE 的 NumPy 实现（Transformer 与 LLM 04）：验证相对性，波长表，PI / NTK-aware / YaRN 三种缩放。
https://arganzheng.life/positional-encoding-and-long-context.html
"""
import numpy as np

def rope_inv_freq(head_dim, base=10000.0):
    i = np.arange(0, head_dim // 2)
    return base ** (-2.0 * i / head_dim)            # theta_i, [head_dim/2]

def rope_cos_sin(positions, head_dim, base=10000.0, inv_freq=None):
    if inv_freq is None:
        inv_freq = rope_inv_freq(head_dim, base)
    angles = np.outer(positions, inv_freq)           # [T, head_dim/2]
    emb = np.concatenate([angles, angles], axis=-1)  # [T, head_dim]，前后半各一份
    return np.cos(emb), np.sin(emb)

def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)

def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin

rng = np.random.default_rng(0)
d_head = 128
q = rng.standard_normal(d_head)
k = rng.standard_normal(d_head)

def score(m, n, base=10000.0):
    cos, sin = rope_cos_sin(np.array([m, n]), d_head, base)
    return apply_rope(q, cos[0], sin[0]) @ apply_rope(k, cos[1], sin[1])

s1 = score(100, 37)
s2 = score(100 + 5000, 37 + 5000)     # 同样的相对距离 63，整体平移 5000
print(f"q_100  . k_37   = {s1:+.6f}")
print(f"q_5100 . k_5037 = {s2:+.6f}   diff = {abs(s1 - s2):.1e}")
print(f"q_100  . k_38   = {score(100, 38):+.6f}   (相对距离 62，应当不同)")

# 与复数闭式 Re[sum q_i conj(k_i) e^{i (m-n) theta_i}] 对照
inv_freq = rope_inv_freq(d_head)
qc = q[:64] + 1j * q[64:]
kc = k[:64] + 1j * k[64:]
closed = np.real(np.sum(qc * np.conj(kc) * np.exp(1j * (100 - 37) * inv_freq)))
print(f"closed form     = {closed:+.6f}")

d_head, L, factor = 128, 8192, 4
base = 10000.0
theta = rope_inv_freq(d_head, base)
lam = 2 * np.pi / theta

# Position Interpolation：所有频率除以 factor
pi_theta = theta / factor

# NTK-aware：改 base
ntk_base = base * factor ** (d_head / (d_head - 2))
ntk_theta = rope_inv_freq(d_head, ntk_base)

# YaRN：按 r = L / lambda 分三段（alpha=1, beta=32）
alpha, beta = 1.0, 32.0
r = L / lam
ramp = np.clip((r - alpha) / (beta - alpha), 0.0, 1.0)    # 0: 全插值, 1: 不动
yarn_theta = (1 - ramp) * theta / factor + ramp * theta
yarn_attn_factor = 0.1 * np.log(factor) + 1                # sqrt(1/t)，乘到 cos/sin 上

print(f"NTK base' = {ntk_base:.0f}, YaRN sqrt(1/t) = {yarn_attn_factor:.4f}")
print(f"{'i':>3} {'lambda':>9} {'r=L/lam':>9} {'theta':>10} {'PI':>10} {'NTK':>10} {'YaRN':>10}")
for i in (0, 8, 16, 24, 32, 40, 48, 56, 63):
    print(f"{i:3d} {lam[i]:9.1f} {r[i]:9.2f} {theta[i]:10.3e} "
          f"{pi_theta[i]:10.3e} {ntk_theta[i]:10.3e} {yarn_theta[i]:10.3e}")
print("YaRN 不动 / 混合 / 全插值 的对数:",
      int((ramp == 1).sum()), int(((ramp > 0) & (ramp < 1)).sum()), int((ramp == 0).sum()))
