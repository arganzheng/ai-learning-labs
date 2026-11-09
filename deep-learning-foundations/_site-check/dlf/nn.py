"""Post 01: Linear / ReLU / softmax-CE forward+backward, gradient check, FLOP counting.
Everything later in the series builds on these ~80 lines."""
import time
import numpy as np

FLOPS = {"fwd": 0, "bwd": 0}
_phase = "fwd"

def reset_flops():
    FLOPS["fwd"] = FLOPS["bwd"] = 0

class Linear:
    def __init__(self, n_in, n_out, rng, std=None, dtype=np.float32):
        std = std if std is not None else np.sqrt(2.0 / n_in)          # Kaiming (ReLU) by default
        self.W = (rng.standard_normal((n_in, n_out)) * std).astype(dtype)
        self.b = np.zeros(n_out, dtype)
        self.dW = np.zeros_like(self.W); self.db = np.zeros_like(self.b)
    def forward(self, X):
        self.X = X                                                     # saved activation for backward
        FLOPS[_phase] += 2 * X.shape[0] * self.W.shape[0] * self.W.shape[1]
        return X @ self.W + self.b
    def backward(self, dY):
        m, k = self.X.shape; n = self.W.shape[1]
        self.dW[...] = self.X.T @ dY                                   # [k,m]@[m,n]  2mkn
        self.db[...] = dY.sum(0)
        dX = dY @ self.W.T                                             # [m,n]@[n,k]  2mkn
        FLOPS[_phase] += 2 * (2 * m * k * n)
        return dX
    def params(self): return [(self.W, self.dW), (self.b, self.db)]

class ReLU:
    def forward(self, X): self.mask = X > 0; return X * self.mask
    def backward(self, dY): return dY * self.mask
    def params(self): return []

def softmax_ce(logits, y):
    """returns loss (mean), dlogits (already divided by batch), probs"""
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    m = logits.shape[0]
    loss = -np.log(p[np.arange(m), y] + 1e-12).mean()
    d = p.copy(); d[np.arange(m), y] -= 1.0                            # p - y
    return loss, d / m, p

class MLP:
    def __init__(self, sizes, rng, dtype=np.float32):
        self.layers = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.layers.append(Linear(a, b, rng, dtype=dtype))
            if i < len(sizes) - 2: self.layers.append(ReLU())
    def forward(self, X):
        global _phase; _phase = "fwd"
        for l in self.layers: X = l.forward(X)
        return X
    def backward(self, d):
        global _phase; _phase = "bwd"
        for l in reversed(self.layers): d = l.backward(d)
        return d
    def params(self):
        return [p for l in self.layers for p in l.params()]

def grad_check(rng, n_checks=30):
    """float64, tiny net, central differences; returns the worst relative error"""
    net = MLP([20, 16, 5], rng, dtype=np.float64)
    X = rng.standard_normal((7, 20)); y = rng.integers(0, 5, 7)
    loss, d, _ = softmax_ce(net.forward(X), y); net.backward(d)
    eps = 1e-6; worst = 0.0
    for P, dP in net.params():
        for _ in range(n_checks):
            idx = tuple(rng.integers(0, s) for s in P.shape)
            old = P[idx]
            P[idx] = old + eps; lp, _, _ = softmax_ce(net.forward(X), y)
            P[idx] = old - eps; lm, _, _ = softmax_ce(net.forward(X), y)
            P[idx] = old
            num = (lp - lm) / (2 * eps); ana = dP[idx]
            rel = abs(num - ana) / max(abs(num) + abs(ana), 1e-12)
            worst = max(worst, rel)
    return worst

def train(net, X, y, Xt, yt, epochs=3, bs=128, lr=0.1, rng=None, log=True, log_epochs=None):
    """plain SGD; prints one line per epoch (or only the epochs in log_epochs)"""
    n = X.shape[0]; acc = float("nan")
    for ep in range(epochs):
        perm = rng.permutation(n); t0 = time.time(); tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss, d, _ = softmax_ce(net.forward(X[idx]), y[idx]); tot += loss * len(idx)
            net.backward(d)
            for P, dP in net.params(): P -= lr * dP
        acc = (net.forward(Xt).argmax(1) == yt).mean()
        if log and (log_epochs is None or ep + 1 in log_epochs):
            print(f"epoch {ep+1:<3d} train loss {tot/n:.4f}  test acc {acc*100:.2f}%  ({time.time()-t0:.1f}s)")
    return acc
