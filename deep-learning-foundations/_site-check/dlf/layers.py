"""Posts 02 and 04: LayerNorm / RMSNorm / Residual (02), Dropout / Embedding (04), and the
64-layer MLP factory used to compare initialization, normalization and residual placements."""
import numpy as np
from . import nn

class LayerNorm:
    def __init__(self, d, eps=1e-5):
        self.g = np.ones(d, np.float32); self.b = np.zeros(d, np.float32)
        self.dg = np.zeros_like(self.g); self.db = np.zeros_like(self.b); self.eps = eps
    def forward(self, X):
        mu = X.mean(-1, keepdims=True); var = X.var(-1, keepdims=True)
        self.r = 1.0 / np.sqrt(var + self.eps); self.xh = (X - mu) * self.r
        return self.xh * self.g + self.b
    def backward(self, dY):
        self.dg[...] = (dY * self.xh).sum(0); self.db[...] = dY.sum(0)
        dxh = dY * self.g
        return self.r * (dxh - dxh.mean(-1, keepdims=True) - self.xh * (dxh * self.xh).mean(-1, keepdims=True))
    def params(self): return [(self.g, self.dg), (self.b, self.db)]

class RMSNorm:
    def __init__(self, d, eps=1e-5):
        self.g = np.ones(d, np.float32); self.dg = np.zeros_like(self.g); self.eps = eps
    def forward(self, X):
        self.r = 1.0 / np.sqrt((X * X).mean(-1, keepdims=True) + self.eps); self.xh = X * self.r
        return self.xh * self.g
    def backward(self, dY):
        self.dg[...] = (dY * self.xh).sum(0)
        dxh = dY * self.g
        return self.r * (dxh - self.xh * (dxh * self.xh).mean(-1, keepdims=True))
    def params(self): return [(self.g, self.dg)]

class Sequential:
    def __init__(self, layers): self.layers = layers
    def forward(self, X):
        for l in self.layers: X = l.forward(X)
        return X
    def backward(self, d):
        for l in reversed(self.layers): d = l.backward(d)
        return d
    def params(self): return [p for l in self.layers for p in l.params()]
    def set_train(self, flag):
        for l in self.layers:
            if isinstance(l, Dropout): l.train = flag

class Residual:
    """h + f(h); f is a Sequential"""
    def __init__(self, f): self.f = f
    def forward(self, X): return X + self.f.forward(X)
    def backward(self, dY): return dY + self.f.backward(dY)
    def params(self): return self.f.params()

class Dropout:
    def __init__(self, p, seed=None): self.p = p; self.train = True; self.rng = np.random.default_rng(seed)
    def forward(self, X):
        if not self.train or self.p == 0: return X
        self.mask = (self.rng.random(X.shape) >= self.p) / (1 - self.p)   # inverted dropout
        return X * self.mask
    def backward(self, dY): return dY * self.mask if self.train and self.p else dY
    def params(self): return []

class Embedding:
    def __init__(self, V, d, rng):
        self.W = (rng.standard_normal((V, d)) * 0.1).astype(np.float32); self.dW = np.zeros_like(self.W)
    def forward(self, idx):                     # idx [m, T] -> [m, T*d]
        self.idx = idx; m, T = idx.shape
        return self.W[idx].reshape(m, -1)
    def backward(self, dY):
        m, T = self.idx.shape; d = self.W.shape[1]
        self.dW[...] = 0; np.add.at(self.dW, self.idx.reshape(-1), dY.reshape(m * T, d))
        return None
    def params(self): return [(self.W, self.dW)]

DEEP_CONFIGS = ["plain-naive", "plain-kaiming", "ln", "res-nonorm", "res-nonorm-scaled", "prenorm", "postnorm"]

def make_deep_mlp(cfg, d=256, L=64, rng=None):
    """784 -> L blocks of width d -> 10, wired according to cfg (see DEEP_CONFIGS)"""
    blocks = [nn.Linear(784, d, rng)]
    for l in range(L):
        if cfg == "plain-naive":       # std = 1/sqrt(n): fine for linear, wrong for ReLU
            blocks += [nn.Linear(d, d, rng, std=1/np.sqrt(d)), nn.ReLU()]
        elif cfg == "plain-kaiming":
            blocks += [nn.Linear(d, d, rng), nn.ReLU()]
        elif cfg == "ln":              # Kaiming + LayerNorm, no residual
            blocks += [nn.Linear(d, d, rng), LayerNorm(d), nn.ReLU()]
        elif cfg == "res-nonorm":      # residual, no norm, no scaling
            blocks += [Residual(Sequential([nn.Linear(d, d, rng), nn.ReLU(), nn.Linear(d, d, rng, std=np.sqrt(1/d))]))]
        elif cfg == "res-nonorm-scaled":  # GPT-2 style: out-proj /sqrt(2L)
            blocks += [Residual(Sequential([nn.Linear(d, d, rng), nn.ReLU(), nn.Linear(d, d, rng, std=np.sqrt(1/d)/np.sqrt(2*L))]))]
        elif cfg == "prenorm":         # h + W2 relu(W1 LN(h))
            blocks += [Residual(Sequential([LayerNorm(d), nn.Linear(d, d, rng), nn.ReLU(), nn.Linear(d, d, rng, std=np.sqrt(1/d)/np.sqrt(2*L))]))]
        elif cfg == "postnorm":        # LN(h + W2 relu(W1 h))
            blocks += [Residual(Sequential([nn.Linear(d, d, rng), nn.ReLU(), nn.Linear(d, d, rng, std=np.sqrt(1/d))])), LayerNorm(d)]
        else:
            raise ValueError(cfg)
    if cfg == "prenorm": blocks.append(LayerNorm(d))
    blocks.append(nn.Linear(d, 10, rng, std=np.sqrt(1/d)))
    return Sequential(blocks)
