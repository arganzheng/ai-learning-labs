"""深度学习基础（05）：CNN —— from LeNet to ResNet and ViT. Needs torch (+ torchvision for `resnet`).
https://arganzheng.life/cnn-from-lenet-to-resnet-and-vit.html

Experiments
  matrix   a 3x3 conv on a 6x6 image IS a 16x36 matrix (sparse, weight-shared)
  resnet   ResNet-50 parameter count and FLOPs at 224x224, via forward hooks (needs torchvision)
  patch    ViT patch embedding == conv with kernel = stride = patch; token counts of common ViTs
  deep     plain vs residual conv nets at depth 20 / 56 on MNIST 14x14 (BN everywhere), 3 epochs
           (quick: 1 epoch on 5k images, depth 20 / 56 still both run)  -- this one takes ~10 min in full
"""
import numpy as np
import torch, torch.nn as tnn, torch.nn.functional as F
from dlf import cli, data

EXPS = {"matrix": "conv as a sparse matrix", "resnet": "ResNet-50 params / FLOPs", "patch": "ViT patch embedding == strided conv",
        "deep": "plain vs residual at depth 20 / 56"}

def conv2d_valid(x, k):
    """x [H,W], k [kh,kw] -> [H-kh+1, W-kw+1] (cross-correlation, as in DL)"""
    H, W = x.shape; kh, kw = k.shape
    out = np.zeros((H - kh + 1, W - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = (x[i:i+kh, j:j+kw] * k).sum()
    return out

def conv_as_matrix(k, H, W):
    """the (Ho*Wo) x (H*W) matrix M with conv(x,k).flatten() == M @ x.flatten()"""
    kh, kw = k.shape; Ho, Wo = H - kh + 1, W - kw + 1
    M = np.zeros((Ho * Wo, H * W))
    for i in range(Ho):
        for j in range(Wo):
            for a in range(kh):
                for b in range(kw):
                    M[i * Wo + j, (i + a) * W + (j + b)] = k[a, b]
    return M

def exp_matrix():
    print("== 1. a 3x3 conv on a 6x6 image IS a 16x36 matrix ==")
    rng = np.random.default_rng(0); x = rng.standard_normal((6, 6)); k = rng.standard_normal((3, 3))
    y = conv2d_valid(x, k); M = conv_as_matrix(k, 6, 6)
    print("max |conv - M@x| =", np.abs(y.flatten() - M @ x.flatten()).max())
    print(f"matrix shape {M.shape}, entries {M.size}, nonzero {np.count_nonzero(M)}, distinct free params {k.size}")
    print("row 0 of M (reshaped 6x6):"); print(np.round(M[0].reshape(6, 6), 2))

def count_flops(model, x):
    flops = {"conv": 0, "linear": 0}
    def hook(m, inp, out):
        if isinstance(m, tnn.Conv2d):
            cout, cin_g, kh, kw = m.weight.shape
            flops["conv"] += 2 * out.numel() * cin_g * kh * kw
        elif isinstance(m, tnn.Linear):
            flops["linear"] += 2 * out.numel() // out.shape[0] * m.in_features * out.shape[0]
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (tnn.Conv2d, tnn.Linear))]
    with torch.no_grad(): model(x)
    for h in hs: h.remove()
    return flops

def exp_resnet():
    print("\n== 2. ResNet-50 (torchvision, no pretrained weights): params and FLOPs at 224x224 ==")
    try: import torchvision
    except ImportError: print("torchvision not installed, skipping"); return
    m = torchvision.models.resnet50(weights=None).eval()
    P = sum(p.numel() for p in m.parameters()); Pconv = sum(p.numel() for n, p in m.named_parameters() if "conv" in n or "downsample.0" in n)
    Pfc = sum(p.numel() for n, p in m.named_parameters() if n.startswith("fc"))
    fl = count_flops(m, torch.zeros(1, 3, 224, 224))
    print(f"params {P/1e6:.2f}M  (conv {Pconv/1e6:.2f}M, fc {Pfc/1e6:.2f}M, bn {(P-Pconv-Pfc)/1e6:.2f}M)")
    print(f"FLOPs per image: conv {fl['conv']/1e9:.2f} G, fc {fl['linear']/1e9:.3f} G  -> {(fl['conv']+fl['linear'])/1e9:.2f} GFLOPs (= {(fl['conv']+fl['linear'])/2e9:.2f} GMACs)")
    print(f"one 3x3 conv 64->64 @56x56: params {64*64*9:,}  FLOPs {2*56*56*64*64*9/1e6:.0f} MFLOPs; as a dense layer on the same tensor it would be ({56*56*64:,})^2 = {(56*56*64)**2/1e9:.1f}G params")
    print("receptive field of L stacked 3x3 (stride 1): 1+2L ->", {L: 1 + 2 * L for L in (1, 5, 10, 20)})

class ConvNet(tnn.Module):
    """stem (stride 2 -> 14x14) + L blocks of [3x3 conv, BN], plain or residual, + GAP + linear"""
    def __init__(self, L=20, c=16, residual=False):
        super().__init__(); self.res = residual
        self.stem = tnn.Sequential(tnn.Conv2d(1, c, 3, stride=2, padding=1, bias=False), tnn.BatchNorm2d(c))
        self.blocks = tnn.ModuleList([tnn.Sequential(tnn.Conv2d(c, c, 3, padding=1, bias=False), tnn.BatchNorm2d(c)) for _ in range(L)])
        self.head = tnn.Linear(c, 10)
    def forward(self, x):
        h = F.relu(self.stem(x))
        for b in self.blocks:
            out = b(h)
            h = F.relu(h + out) if self.res else F.relu(out)
        return self.head(h.mean((2, 3)))

def grad_norm_first_last(m, xb, yb):
    m.zero_grad(); F.cross_entropy(m(xb), yb).backward()
    g = lambda blk: blk[0].weight.grad.norm().item()
    return g(m.blocks[0]), g(m.blocks[-1])

def exp_deep(quick):
    n, epochs = (5000, 1) if quick else (20000, 3)
    print(f"\n== 3. deep conv net (BN everywhere) on MNIST 14x14, {epochs} epochs of {n//1000}k images, SGD momentum lr 0.05, clip 1.0: plain vs residual at L=20 / 56 ==")
    X, y = data.load_mnist("train"); Xt, yt = data.load_mnist("test")
    X = torch.tensor(X[:n]).view(-1, 1, 28, 28); y = torch.tensor(y[:n]); Xt = torch.tensor(Xt[:5000]).view(-1, 1, 28, 28); yt = torch.tensor(yt[:5000])
    X = (X - 0.13) / 0.31; Xt = (Xt - 0.13) / 0.31
    for L in (20, 56):
        for res in (False, True):
            torch.manual_seed(0); m = ConvNet(L, 16, res); opt = torch.optim.SGD(m.parameters(), 0.05, momentum=0.9)
            g0, gL = grad_norm_first_last(m, X[:100], y[:100])
            t = cli.Timer(); losses = []
            for ep in range(epochs):
                perm = torch.randperm(n)
                for i in range(0, n, 100):
                    idx = perm[i:i+100]; loss = F.cross_entropy(m(X[idx]), y[idx]); opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); losses.append(loss.item())
            with torch.no_grad():                                   # precise BN: re-estimate running stats on training data
                for mod in m.modules():
                    if isinstance(mod, tnn.BatchNorm2d): mod.reset_running_stats(); mod.momentum = None
                for i in range(0, min(n, 10000), 500): m(X[i:i+500])
            m.eval()
            with torch.no_grad(): acc = (m(Xt).argmax(1) == yt).float().mean().item()
            e1 = min(150, len(losses) // 2); ep1 = np.mean(losses[e1:e1 + 50])   # end of epoch 1 (steps 150-200 in the full run)
            print(f"L={L:2d} {'residual' if res else 'plain   '}: init grad norm block1 {g0:.1e} vs block{L} {gL:.1e} (ratio {g0/gL:.2f}) | train loss @ep1 {ep1:.3f} @end {np.mean(losses[-50:]):.3f} | test acc {acc*100:5.1f}%  ({t})")

def exp_patch():
    print("\n== 4. ViT patch embedding == conv with kernel = stride = patch ==")
    torch.manual_seed(0); B, C, H, W, p, d = 2, 3, 224, 224, 16, 768
    x = torch.randn(B, C, H, W)
    conv = tnn.Conv2d(C, d, kernel_size=p, stride=p)
    tokens_conv = conv(x).flatten(2).transpose(1, 2)                             # [B, N, d]
    patches = x.unfold(2, p, p).unfold(3, p, p)                                 # [B, C, H/p, W/p, p, p]
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * p * p)       # [B, N, C*p*p]
    tokens_lin = patches @ conv.weight.view(d, -1).T + conv.bias
    print("max |conv - unfold+linear| =", (tokens_conv - tokens_lin).abs().max().item())
    print(f"tokens per image: {tokens_conv.shape[1]}  (= (224/16)^2);  patch embedding params {sum(p.numel() for p in conv.parameters()):,}  (= 3*16*16*768 + 768)")
    for name, res, pp in [("ViT-B/16 @224", 224, 16), ("ViT-L/14 @224", 224, 14), ("CLIP ViT-L/14 @336", 336, 14), ("ViT-H/14 @224", 224, 14)]:
        print(f"  {name:20s} -> {(res//pp)**2} tokens")

if __name__ == "__main__":
    which, quick = cli.parse(EXPS, __doc__)
    if "matrix" in which: exp_matrix()
    if "resnet" in which: exp_resnet()
    if "patch" in which: exp_patch()
    if "deep" in which: exp_deep(quick)
