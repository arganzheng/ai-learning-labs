"""Post 03: SGD / Momentum / Nesterov, Adam / AdamW (decoupled vs L2), warmup + cosine, clipping."""
import numpy as np
from . import nn

class SGD:
    def __init__(self, params, lr, momentum=0.0, nesterov=False):
        self.p = params; self.lr = lr; self.mom = momentum; self.nest = nesterov
        self.v = [np.zeros_like(P) for P, _ in params]
    def step(self):
        for (P, g), v in zip(self.p, self.v):
            if self.mom:
                v *= self.mom; v += g
                P -= self.lr * (g + self.mom * v if self.nest else v)
            else:
                P -= self.lr * g
    def state_bytes(self): return sum(v.nbytes for v in self.v) if self.mom else 0

class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, decoupled=True, bias_correction=True):
        self.p = params; self.lr = lr; self.b1, self.b2, self.eps = b1, b2, eps
        self.wd = wd; self.decoupled = decoupled; self.bc = bias_correction; self.t = 0
        self.m = [np.zeros_like(P) for P, _ in params]; self.v = [np.zeros_like(P) for P, _ in params]
        self.last_update_max = 0.0
    def step(self):
        self.t += 1; b1, b2 = self.b1, self.b2
        c1 = 1 - b1 ** self.t if self.bc else 1.0; c2 = 1 - b2 ** self.t if self.bc else 1.0
        self.last_update_max = 0.0
        for (P, g), m, v in zip(self.p, self.m, self.v):
            if self.wd and not self.decoupled: g = g + self.wd * P          # L2 inside the gradient
            m *= b1; m += (1 - b1) * g
            v *= b2; v += (1 - b2) * g * g
            upd = (m / c1) / (np.sqrt(v / c2) + self.eps)
            if self.wd and self.decoupled: P -= self.lr * self.wd * P      # AdamW
            P -= self.lr * upd
            self.last_update_max = max(self.last_update_max, np.abs(upd).max())
    def state_bytes(self): return sum(a.nbytes for a in self.m + self.v)

def lr_at(step, peak, warmup, total, floor=0.1):
    """linear warmup, then cosine decay to floor*peak"""
    if step < warmup: return peak * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * prog)))

def run(net, opt, X, y, steps, bs, rng, warmup=0, sched=False, clip=None, outlier_at=None):
    """minibatch training loop; returns (losses, grad norms, "ok" | "nan")"""
    peak = opt.lr; losses = []; gnorms = []
    for s in range(steps):
        if sched or warmup: opt.lr = lr_at(s, peak, warmup, steps) if sched else (peak * min(1.0, (s + 1) / warmup))
        idx = rng.integers(0, X.shape[0], bs); xb = X[idx]
        if outlier_at is not None and s == outlier_at: xb = xb * 50.0
        loss, d, _ = nn.softmax_ce(net.forward(xb), y[idx]); net.backward(d)
        gn = np.sqrt(sum((g * g).sum() for _, g in net.params()))
        if clip and gn > clip:
            for _, g in net.params(): g *= clip / gn
        if not np.isfinite(loss): return losses, gnorms, "nan"
        opt.step(); losses.append(loss); gnorms.append(gn)
    return losses, gnorms, "ok"

def accuracy(net, Xt, yt): return (net.forward(Xt).argmax(1) == yt).mean()
