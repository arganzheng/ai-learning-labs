"""深度学习基础（03）：优化器 —— from SGD to AdamW.
https://arganzheng.life/optimizers-from-sgd-to-adamw.html

Experiments
  compare   SGD / Momentum / Adam / AdamW on a 2-layer MLP, 1 epoch, best over a small lr grid
  bias      Adam first-step update magnitude with and without bias correction
  warmup    Adam on the 64-layer pre-norm net: beta2 x warmup
  adamw     AdamW vs Adam+L2 at the same lambda: per-layer weight norms after 2 epochs
  scaling   linear lr scaling: lr = 0.05 * B/32 for B in 32..8192 at a fixed sample budget
  clip      gradient clipping vs one outlier batch (inputs x50)
"""
import numpy as np
from dlf import cli, data, nn
from dlf.layers import make_deep_mlp
from dlf.optim import SGD, Adam, run, accuracy

EXPS = {"compare": "optimizer comparison, 1 epoch", "bias": "Adam bias correction", "warmup": "warmup on a 64-layer net",
        "adamw": "AdamW vs Adam+L2 weight norms", "scaling": "batch-size / lr linear scaling", "clip": "clipping vs an outlier batch"}

def new_net():
    rng = np.random.default_rng(0); return rng, nn.MLP([784, 256, 10], rng)

def exp_compare(X, y, Xt, yt, quick):
    steps = 150 if quick else 469
    print(f"== 1. optimizer comparison, 2-layer MLP, {steps} steps of batch 128, best over lr grid ==")
    grid = {"SGD": [0.03, 0.1, 0.3], "Momentum0.9": [0.003, 0.01, 0.03], "Adam": [3e-4, 1e-3, 3e-3], "AdamW(wd0.1)": [3e-4, 1e-3, 3e-3]}
    if quick: grid = {k: v[1:2] for k, v in grid.items()}
    for name, lrs in grid.items():
        best = None
        for lr in lrs:
            rng, net = new_net()
            opt = {"SGD": lambda: SGD(net.params(), lr), "Momentum0.9": lambda: SGD(net.params(), lr, 0.9),
                   "Adam": lambda: Adam(net.params(), lr), "AdamW(wd0.1)": lambda: Adam(net.params(), lr, wd=0.1)}[name]()
            l, _, st = run(net, opt, X, y, steps, 128, rng)
            a = accuracy(net, Xt, yt) if st == "ok" else float("nan")
            if best is None or (st == "ok" and a > best[1]): best = (lr, a, np.mean(l[-50:]), opt.state_bytes())
        print(f"{name:14s} best lr {best[0]:<7g} test acc {best[1]*100:5.2f}%  final loss {best[2]:.3f}  state {best[3]/1024:.0f} KiB (params {sum(P.nbytes for P,_ in net.params())/1024:.0f} KiB)")

def exp_bias(X, y):
    print("\n== 2. Adam first-step update magnitude (|update| / lr), with and without bias correction ==")
    for b2 in (0.999, 0.95):
        for bc in (True, False):
            rng, net = new_net(); opt = Adam(net.params(), 1e-3, b2=b2, bias_correction=bc)
            run(net, opt, X, y, 1, 128, rng)
            print(f"beta2 {b2}  bias_correction {str(bc):5s}  max |update|/lr at step 1: {opt.last_update_max:8.2f}   (expected {1/np.sqrt(1-b2):.1f} without correction)")

def exp_warmup(X, y, Xt, yt, quick):
    steps = 150 if quick else 400
    print(f"\n== 3. warmup on the 64-layer pre-norm net, Adam lr 1e-3, {steps} steps ==")
    for b2 in (0.999, 0.95):
        for wu in (0, 100):
            rng = np.random.default_rng(0); net = make_deep_mlp("prenorm", rng=rng)
            opt = Adam(net.params(), 1e-3, b2=b2)
            l, g, st = run(net, opt, X, y, steps, 128, rng, warmup=wu)
            tag = f"beta2 {b2} warmup {wu:3d}"
            if st == "nan": print(f"{tag}: NaN at step {len(l)}"); continue
            print(f"{tag}: loss @20 {l[19]:.3f}  @100 {l[99]:.3f}  @{steps} {l[-1]:.3f}  max grad norm in first 50 steps {max(g[:50]):.1f}  acc {accuracy(net,Xt,yt)*100:.1f}%")

def exp_adamw(X, y, Xt, yt, quick):
    steps = 300 if quick else 938
    print(f"\n== 4. AdamW vs Adam+L2 (same lambda=0.1, lr 1e-3, {steps} steps): per-layer weight norms ==")
    for decoupled in (True, False):
        rng, net = new_net(); opt = Adam(net.params(), 1e-3, wd=0.1, decoupled=decoupled)
        run(net, opt, X, y, steps, 128, rng)
        norms = [np.linalg.norm(P) for P, _ in net.params() if P.ndim == 2]
        gr = [np.sqrt(v).mean() for v, (P, _) in zip(opt.v, net.params()) if P.ndim == 2]
        print(f"{'AdamW' if decoupled else 'Adam+L2':8s} ||W1|| {norms[0]:6.2f}  ||W2|| {norms[1]:6.2f}   mean sqrt(v): W1 {gr[0]:.1e}  W2 {gr[1]:.1e}   acc {accuracy(net,Xt,yt)*100:.2f}%")
    _, net = new_net()
    print(f"{'init':8s} ||W1|| {np.linalg.norm(net.layers[0].W):6.2f}  ||W2|| {np.linalg.norm(net.layers[2].W):6.2f}")

def exp_scaling(X, y, Xt, yt, quick):
    budget = 60000 if quick else 240000
    print(f"\n== 5. linear lr scaling: SGD, lr = 0.05 * B/32, fixed budget of {budget:,} samples ==")
    for B in (32, 128, 512, 2048, 8192):
        rng, net = new_net(); lr = 0.05 * B / 32; steps = budget // B
        opt = SGD(net.params(), lr); t = cli.Timer()
        l, _, st = run(net, opt, X, y, steps, B, rng)
        print(f"B {B:5d} lr {lr:6.2f} steps {steps:5d}: " + (f"NaN at step {len(l)}" if st == "nan" else f"final loss {np.mean(l[-max(1,len(l)//20):]):.3f}  acc {accuracy(net,Xt,yt)*100:.2f}%") + f"  ({t})")

def exp_clip(X, y):
    print("\n== 6. gradient clipping vs an outlier batch (inputs x50 at step 200), Adam lr 1e-3, 2-layer ==")
    for clip in (None, 1.0):
        rng, net = new_net(); opt = Adam(net.params(), 1e-3)
        l, g, st = run(net, opt, X, y, 600, 128, rng, clip=clip, outlier_at=200)
        print(f"clip {str(clip):5s}: grad norm @199 {g[198]:.2f} @200 {g[200]:.1f}   loss @199 {l[198]:.3f} @201 {l[200]:.3f} @210 {l[209]:.3f} @250 {l[249]:.3f} @600 {l[-1]:.3f}")

if __name__ == "__main__":
    which, quick = cli.parse(EXPS, __doc__)
    X, y, Xt, yt = data.load_mnist_standardized()
    Xt, yt = Xt[:5000], yt[:5000]
    if "compare" in which: exp_compare(X, y, Xt, yt, quick)
    if "bias" in which: exp_bias(X, y)
    if "warmup" in which: exp_warmup(X, y, Xt, yt, quick)
    if "adamw" in which: exp_adamw(X, y, Xt, yt, quick)
    if "scaling" in which: exp_scaling(X, y, Xt, yt, quick)
    if "clip" in which: exp_clip(X, y)
