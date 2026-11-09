"""深度学习基础（01）：手推反向传播 —— gradient check, FLOPs fwd:bwd = 1:2, saved activations, MNIST 97%.
https://arganzheng.life/backpropagation-by-hand.html

Experiments
  check   gradient check against central differences (float64)
  flops   forward / backward FLOP count and saved-activation bytes for one batch
  torch   same weights + batch through PyTorch autograd, compare gradients (needs torch)
  train   plain SGD on MNIST, 15 epochs (quick: 3)
"""
import numpy as np
from dlf import cli, data, nn

EXPS = {"check": "gradient check vs finite differences", "flops": "fwd/bwd FLOPs + saved activations",
        "torch": "compare with PyTorch autograd", "train": "SGD on MNIST"}

def exp_check(rng):
    print("== gradient check: [20,16,5] float64 net, 30 random entries per parameter, eps 1e-6 ==")
    print(f"grad check worst rel err: {nn.grad_check(rng):.2e}")

def exp_flops(net, X, y):
    print("\n== FLOPs and activation memory for one step, batch 128 ==")
    nparams = sum(P.size for P, _ in net.params())
    print("params:", nparams)
    nn.reset_flops()
    loss, d, _ = nn.softmax_ce(net.forward(X[:128]), y[:128]); net.backward(d)
    F = nn.FLOPS
    print(f"batch 128: fwd {F['fwd']/1e6:.1f} MFLOPs  bwd {F['bwd']/1e6:.1f} MFLOPs  ratio {F['bwd']/F['fwd']:.2f}")
    print(f"  2*N*tokens = {2*nparams*128/1e6:.1f} MFLOPs (fwd approx ignoring bias)")
    act = sum(l.X.nbytes for l in net.layers if isinstance(l, nn.Linear)) + sum(l.mask.nbytes for l in net.layers if isinstance(l, nn.ReLU))
    print(f"  saved activations: {act/1024:.0f} KiB  vs weights {sum(P.nbytes for P,_ in net.params())/1024:.0f} KiB")

def exp_torch(net, X, y):
    print("\n== same weights and batch through PyTorch autograd ==")
    try: import torch
    except ImportError: print("torch not installed, skipping"); return
    Xb, yb = X[:128], y[:128]
    loss, d, _ = nn.softmax_ce(net.forward(Xb), yb); net.backward(d)
    W1, b1 = net.layers[0].W, net.layers[0].b; W2, b2 = net.layers[2].W, net.layers[2].b
    tW1, tb1, tW2, tb2 = [torch.tensor(a, requires_grad=True) for a in (W1, b1, W2, b2)]
    logits = torch.relu(torch.tensor(Xb) @ tW1 + tb1) @ tW2 + tb2
    tloss = torch.nn.functional.cross_entropy(logits, torch.tensor(yb)); tloss.backward()
    print("loss numpy %.6f torch %.6f" % (loss, tloss.item()))
    for name, ours, theirs in [("dW1", net.layers[0].dW, tW1.grad), ("db1", net.layers[0].db, tb1.grad),
                               ("dW2", net.layers[2].dW, tW2.grad), ("db2", net.layers[2].db, tb2.grad)]:
        print(name, "max abs diff %.2e" % np.abs(ours - theirs.numpy()).max())

def exp_train(net, X, y, Xt, yt, rng, quick):
    epochs = 3 if quick else 15
    print(f"\n== SGD lr 0.1, batch 128, {epochs} epochs ==")
    nn.train(net, X, y, Xt, yt, epochs=epochs, lr=0.1, rng=rng, log_epochs=None if quick else {1, 5, 10, 15})

if __name__ == "__main__":
    which, quick = cli.parse(EXPS, __doc__)
    rng = np.random.default_rng(0)
    if "check" in which: exp_check(rng)
    X, y = data.load_mnist("train"); Xt, yt = data.load_mnist("test")
    net = nn.MLP([784, 256, 10], rng)
    if "flops" in which: exp_flops(net, X, y)
    if "torch" in which: exp_torch(net, X, y)
    if "train" in which: exp_train(net, X, y, Xt, yt, rng, quick)
