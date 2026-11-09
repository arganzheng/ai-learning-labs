"""深度学习基础（04）：正则化与泛化 —— double descent, dropout / weight decay / early stopping, memorization.
https://arganzheng.life/regularization-and-generalization.html

Experiments
  dd     double descent: 4000 MNIST samples with 20% label noise, width sweep 2..2048, 6000 full-batch Adam steps
  reg    1000 samples, width 512, {none, wd, dropout, both}, 300 epochs: train / test loss and best early-stopping point
  lm     char-level MLP LM on Python's stdlib source, 20K / 200K / 2M training chars: held-out loss per epoch
         and a memorization probe (greedy-generate 32 chars from a training prefix, count exact reproductions)
Full run takes ~15 min on a laptop CPU; --quick cuts every sweep to a couple of minutes.
"""
import numpy as np
from dlf import cli, data, nn
from dlf.layers import Dropout, Embedding, Sequential
from dlf.optim import Adam

EXPS = {"dd": "double descent width sweep", "reg": "regularizers on 1000 samples", "lm": "char LM: data size vs epochs"}

def ce_eval(net, X, y, bs=4096):
    net.set_train(False); tot = 0.0; correct = 0
    for i in range(0, len(X), bs):
        logits = net.forward(X[i:i+bs]); loss, _, p = nn.softmax_ce(logits, y[i:i+bs])
        tot += loss * len(logits); correct += (logits.argmax(1) == y[i:i+bs]).sum()
    net.set_train(True); return tot / len(X), correct / len(X)

def exp_double_descent(X, y, Xt, yt, quick):
    rng = np.random.default_rng(0)
    n = 4000; idx = rng.permutation(len(X))[:n]; Xs, ys = X[idx], y[idx].copy()
    noisy = rng.random(n) < 0.2; ys[noisy] = rng.integers(0, 10, noisy.sum())      # 20% label noise
    steps = 1500 if quick else 6000
    widths = [2, 8, 16, 64, 512] if quick else [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    print(f"== 1. double descent: n={n} train, 20% label noise, 2-layer MLP, width sweep, Adam 1e-3, {steps} full-batch steps ==")
    print(f"{'width':>6} {'params':>8} {'params/n':>8} {'train err':>9} {'test err':>8} {'test loss':>9}")
    for w in widths:
        rng2 = np.random.default_rng(1); net = Sequential([nn.Linear(784, w, rng2), nn.ReLU(), nn.Linear(w, 10, rng2, std=np.sqrt(1/w))])
        opt = Adam(net.params(), 1e-3)
        for s in range(steps):
            loss, d, _ = nn.softmax_ce(net.forward(Xs), ys); net.backward(d); opt.step()
        trl, tra = ce_eval(net, Xs, ys); tel, tea = ce_eval(net, Xt, yt)
        P = sum(p.size for p, _ in net.params())
        print(f"{w:6d} {P:8d} {P/n:8.1f} {100*(1-tra):8.1f}% {100*(1-tea):7.1f}% {tel:9.3f}")

def exp_regularizers(X, y, Xt, yt, quick):
    rng = np.random.default_rng(0); n = 1000; idx = rng.permutation(len(X))[:n]; Xs, ys = X[idx], y[idx]
    epochs = 60 if quick else 300; marks = {10, 30, epochs} if quick else {10, 30, 100, 300}
    print(f"\n== 2. regularizers: n={n}, width 512, AdamW 1e-3, {epochs} epochs (batch 100): train/test loss ==")
    cfgs = {"none": (0.0, 0.0), "wd=0.5": (0.5, 0.0), "dropout=0.5": (0.0, 0.5), "wd=0.5+dropout=0.5": (0.5, 0.5)}
    for name, (wd, p) in cfgs.items():
        rng2 = np.random.default_rng(1)
        net = Sequential([nn.Linear(784, 512, rng2), nn.ReLU(), Dropout(p, seed=2), nn.Linear(512, 10, rng2, std=np.sqrt(1/512))])
        opt = Adam(net.params(), 1e-3, wd=wd); best = (9, 0, 0)
        for ep in range(epochs):
            perm = rng2.permutation(n)
            for i in range(0, n, 100):
                b = perm[i:i+100]; loss, d, _ = nn.softmax_ce(net.forward(Xs[b]), ys[b]); net.backward(d); opt.step()
            if (ep + 1) % 10 == 0:
                tel, tea = ce_eval(net, Xt, yt)
                if tel < best[0]: best = (tel, tea, ep + 1)
                if ep + 1 in marks:
                    trl, tra = ce_eval(net, Xs, ys)
                    print(f"{name:20s} ep {ep+1:3d}  train loss {trl:.3f}  test loss {tel:.3f}  test acc {tea*100:.1f}%")
        print(f"{name:20s} early stopping: best test loss {best[0]:.3f} (acc {best[1]*100:.1f}%) at epoch {best[2]}")

def exp_lm(quick, T=8, d=16, hid=512):
    print(f"\n== 3. char-level MLP LM (context {T}) on Python stdlib source: train vs held-out loss per epoch ==")
    runs = [(20_000, 8)] if quick else [(20_000, 40), (200_000, 10), (2_000_000, 10)]
    probe_at = {1, 2, 4, 6, 8, 10, 12, 16, 20, 30, 40}
    for size, epochs in runs:
        data_ids, V = data.load_text(size + 50_000)
        tr, va = data_ids[:size], data_ids[size:size + 50_000]
        def batches(arr, bs, rng):
            starts = rng.permutation(len(arr) - T - 1)
            for i in range(0, len(starts) - bs, bs):
                s = starts[i:i+bs]; ctx = np.stack([arr[j:j+T] for j in s]); yield ctx, arr[s + T]
        def evaluate(net, arr, rng, nb=40):
            tot = 0.0; k = 0
            for ctx, yb in batches(arr, 1024, rng):
                loss, _, _ = nn.softmax_ce(net.forward(ctx), yb); tot += loss; k += 1
                if k >= nb: break
            return tot / k
        def memorized(net, arr, rng, n=200, gen=32):
            """fraction of training positions where greedy generation of `gen` chars from a T-char prefix reproduces the text exactly"""
            starts = rng.integers(0, len(arr) - T - gen, n); hits = 0
            for s in starts:
                ctx = arr[s:s+T].copy(); ok = True
                for g in range(gen):
                    nxt = net.forward(ctx[None, :]).argmax()
                    if nxt != arr[s + T + g]: ok = False; break
                    ctx = np.append(ctx[1:], nxt)
                hits += ok
            return hits / n
        rng = np.random.default_rng(0)
        net = Sequential([Embedding(V, d, rng), nn.Linear(T * d, hid, rng), nn.ReLU(), nn.Linear(hid, V, rng, std=np.sqrt(1/hid))])
        opt = Adam(net.params(), 1e-3); P = sum(p.size for p, _ in net.params())
        print(f"-- train chars {size:,}  vocab {V}  params {P:,}  ({P/size:.2f} params per training char)")
        t = cli.Timer()
        for ep in range(1, epochs + 1):
            for ctx, yb in batches(tr, 512, rng):
                loss, dlog, _ = nn.softmax_ce(net.forward(ctx), yb); net.backward(dlog); opt.step()
            trl = evaluate(net, tr, np.random.default_rng(1)); val = evaluate(net, va, np.random.default_rng(2))
            mem = memorized(net, tr, np.random.default_rng(3)) if ep in probe_at else float("nan")
            print(f"   epoch {ep:2d}  train loss {trl:.3f}  held-out loss {val:.3f}  gap {val-trl:+.3f}  memorized(32 chars) {mem*100:5.1f}%  ({t})")

if __name__ == "__main__":
    which, quick = cli.parse(EXPS, __doc__)
    if "dd" in which or "reg" in which:
        X, y, Xt, yt = data.load_mnist_standardized()
        if "dd" in which: exp_double_descent(X, y, Xt, yt, quick)
        if "reg" in which: exp_regularizers(X, y, Xt, yt, quick)
    if "lm" in which: exp_lm(quick)
