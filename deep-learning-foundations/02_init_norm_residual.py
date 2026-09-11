"""深度学习基础（02）：初始化、归一化与残差 —— why a 64-layer MLP does or does not train.
https://arganzheng.life/initialization-normalization-and-residual.html

Seven wirings of the same 64 x 256 MLP (see dlf.layers.DEEP_CONFIGS):
  plain-naive         std 1/sqrt(n)                (activations vanish)
  plain-kaiming       std sqrt(2/n)                (fine at init, still hard to train)
  ln                  Kaiming + LayerNorm, no residual
  res-nonorm          residual, no norm             (activation std explodes to ~1e9)
  res-nonorm-scaled   residual, out-proj / sqrt(2L) (GPT-2 trick; fixes init only)
  prenorm             h + f(LN(h))
  postnorm            LN(h + f(h))

Experiments
  init    activation std and gradient norm per depth at initialization
  train   300 SGD steps at lr 0.05 and 0.005 (quick: 100 steps, lr 0.05 only)
"""
import numpy as np
from dlf import cli, data, nn
from dlf.layers import DEEP_CONFIGS, make_deep_mlp

EXPS = {"init": "activation / gradient statistics at init", "train": "300 SGD steps per config"}

def stats(net, X, y):
    """per-block activation std (forward) and per-Linear grad norm (backward), at init"""
    acts = []; h = X
    for l in net.layers:
        h = l.forward(h)
        if not isinstance(l, nn.ReLU): acts.append(h.std())
    loss, d, _ = nn.softmax_ce(h, y); net.backward(d)
    gn = [np.linalg.norm(dP) for P, dP in net.params() if P.ndim == 2]
    return loss, acts, gn

def train_steps(net, X, y, steps, bs=128, lr=0.05, rng=None):
    losses = []
    for s in range(steps):
        idx = rng.integers(0, X.shape[0], bs)
        loss, d, _ = nn.softmax_ce(net.forward(X[idx]), y[idx]); net.backward(d)
        if not np.isfinite(loss): return losses, "nan"
        for P, dP in net.params(): P -= lr * dP
        losses.append(loss)
    return losses, "ok"

def exp_init(X, y):
    print("0.9^128 = %.2e   1.1^128 = %.2e" % (0.9**128, 1.1**128))
    Xb, yb = X[:256], y[:256]; pick = [0, 7, 15, 31, 63]
    print("\n== activation std at blocks 1/8/16/32/64 (init), then grad norm of the Linear at those depths ==")
    for c in DEEP_CONFIGS:
        rng = np.random.default_rng(0); net = make_deep_mlp(c, rng=rng)
        loss, acts, gn = stats(net, Xb, yb)
        a = [acts[min(i + 1, len(acts) - 1)] for i in pick]
        stride = 2 if c.startswith("res") or c in ("prenorm", "postnorm") else 1   # Linears per block
        g = [gn[min(1 + i * stride, len(gn) - 2)] for i in pick]
        print(f"{c:18s} loss {loss:6.3f} | act std " + " ".join(f"{v:9.2e}" for v in a) + " | grad " + " ".join(f"{v:9.2e}" for v in g))

def exp_train(X, y, Xt, yt, quick):
    steps = 100 if quick else 300
    for lr in ((0.05,) if quick else (0.05, 0.005)):
        print(f"\n== {steps} SGD steps, lr {lr}, batch 128: loss at step 1 / {steps//3} / {steps} ==")
        for c in DEEP_CONFIGS:
            rng = np.random.default_rng(0); net = make_deep_mlp(c, rng=rng); t = cli.Timer()
            losses, st = train_steps(net, X, y, steps, lr=lr, rng=rng)
            if st == "nan": print(f"{c:18s} diverged (NaN) at step {len(losses)}"); continue
            acc = (net.forward(Xt[:2000]).argmax(1) == yt[:2000]).mean()
            print(f"{c:18s} {losses[0]:.3f} / {losses[steps//3 - 1]:.3f} / {losses[-1]:.3f}   test acc {acc*100:5.1f}%  ({t})")

if __name__ == "__main__":
    which, quick = cli.parse(EXPS, __doc__)
    np.seterr(all="ignore")                         # res-nonorm overflows to inf/NaN on purpose; that is the result, not a bug
    X, y, Xt, yt = data.load_mnist_standardized()
    if "init" in which: exp_init(X, y)
    if "train" in which: exp_train(X, y, Xt, yt, quick)
