"""深度学习基础（06）：RNN、LSTM 与 attention 的诞生. Needs torch.
https://arganzheng.life/rnn-lstm-and-the-birth-of-attention.html

Experiments
  bptt     vanilla RNN, backprop through time: ||dL/dh_t|| vs distance from the loss (NumPy, 3 weight scales)
  memory   "output the first token after T steps": RNN vs LSTM for T = 5..80, 1500 Adam steps
  forget   same task, LSTM with both forget-gate bias vectors set to 1 (effective bias 2): T = 20 / 40 / 80, eval every 500 steps up to 6000
  seq2seq  reverse-the-sequence with a GRU encoder-decoder: fixed-vector bottleneck vs Bahdanau attention, T = 8 / 16 / 32,
           plus the learned alignment matrix
  timing   forward wall time vs sequence length: RNN (sequential) vs self-attention (parallel), CPU
Full run ~5 min on a laptop CPU; --quick shortens the training loops.
"""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from dlf import cli

EXPS = {"bptt": "gradient decay through time", "memory": "RNN vs LSTM memory length", "forget": "LSTM forget bias = 1",
        "seq2seq": "bottleneck vs attention", "timing": "RNN vs self-attention wall time"}
THREADS = min(8, torch.get_num_threads())

def exp_bptt():
    print("== 1. vanilla RNN, BPTT: ||dL/dh_t|| as a function of distance from the loss (NumPy) ==")
    rng = np.random.default_rng(0); d = 64; T = 60
    for scale in (0.5, 1.0, 1.5):
        W = rng.standard_normal((d, d)) / np.sqrt(d) * scale          # spectral radius ~ scale
        U = rng.standard_normal((d, d)) / np.sqrt(d)
        x = rng.standard_normal((T, d)); h = np.zeros(d); hs = [h]
        for t in range(T): h = np.tanh(W @ h + U @ x[t]); hs.append(h)
        g = np.ones(d); norms = []                                       # loss only on the last state: L = sum(h_T)
        for t in range(T, 0, -1):
            norms.append(np.linalg.norm(g))
            g = W.T @ (g * (1 - hs[t] ** 2))                             # J_t^T g,  J_t = diag(1-h_t^2) W
        norms = norms[::-1]                                              # index 0 = earliest step
        sv = np.linalg.svd(W, compute_uv=False)[0]
        print(f"scale {scale}: sigma_max(W) {sv:.2f} | ||dL/dh_t|| at distance 0/10/20/40/59: " + " ".join(f"{norms[T-1-k]:.1e}" for k in (0, 10, 20, 40, 59)))

class RNNCls(nn.Module):
    def __init__(self, V, d, kind, forget_bias=None):
        super().__init__(); self.emb = nn.Embedding(V, d)
        self.rnn = (nn.RNN if kind == "rnn" else nn.LSTM)(d, d, batch_first=True); self.head = nn.Linear(d, V)
        if forget_bias is not None:                                      # gates are ordered (i, f, g, o) in PyTorch
            for name in ("bias_ih_l0", "bias_hh_l0"):                    # PyTorch keeps two bias vectors that are summed,
                with torch.no_grad(): getattr(self.rnn, name)[d:2*d].fill_(forget_bias)   # so the effective forget bias is 2x
    def forward(self, x):
        out, _ = self.rnn(self.emb(x)); return self.head(out[:, -1])

def train_memory(m, T, steps, V=10, eval_every=None):
    """returns list of (step, acc) at each evaluation; stops early once acc hits 100%"""
    opt = torch.optim.Adam(m.parameters(), 1e-3); g = torch.Generator().manual_seed(1); hist = []
    for s in range(1, steps + 1):
        x = torch.randint(0, V, (64, T), generator=g); loss = F.cross_entropy(m(x), x[:, 0])
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if s == steps or (eval_every and s % eval_every == 0):
            with torch.no_grad():
                x = torch.randint(0, V, (2000, T), generator=g); acc = (m(x).argmax(1) == x[:, 0]).float().mean().item()
            hist.append((s, acc, loss.item()))
            if acc == 1.0: break
    return hist

def exp_memory(quick):
    steps = 500 if quick else 1500; Ts = (5, 10, 20) if quick else (5, 10, 20, 40, 80)
    print(f"\n== 2. remember the first token and output it after T steps: RNN vs LSTM, {steps} steps Adam 1e-3, batch 64 ==")
    torch.set_num_threads(THREADS)
    for T in Ts:
        row = []
        for kind in ("rnn", "lstm"):
            torch.manual_seed(0); m = RNNCls(10, 64, kind); t = cli.Timer()
            s, acc, loss = train_memory(m, T, steps)[-1]
            row.append(f"{kind.upper():4s} {acc*100:5.1f}% (loss {loss:.2f}, {t})")
        print(f"T={T:3d}: " + "   ".join(row))

def exp_forget(quick):
    steps = 1500 if quick else 6000; Ts = (20,) if quick else (20, 40, 80)
    print(f"\n== 3. LSTM with forget-gate bias = 1 in both bias vectors (b_ih + b_hh = 2), eval every 500 steps up to {steps} ==")
    torch.set_num_threads(THREADS)
    for T in Ts:
        torch.manual_seed(0); m = RNNCls(10, 64, "lstm", forget_bias=1.0); t = cli.Timer()
        hist = train_memory(m, T, steps, eval_every=500)
        s, acc, loss = hist[-1]
        print(f"T={T:3d}: acc {acc*100:5.1f}% at step {s} ({t})   trajectory: " + " ".join(f"{h[0]}:{h[1]*100:.0f}%" for h in hist))

class Seq2Seq(nn.Module):
    """GRU encoder-decoder; attention=False -> decoder conditions only on the final encoder state (fixed-vector bottleneck)"""
    def __init__(self, V, d, attention):
        super().__init__(); self.att = attention; self.emb = nn.Embedding(V + 1, d)   # +1 for <sos>
        self.enc = nn.GRU(d, d, batch_first=True); self.dec = nn.GRUCell(2 * d if attention else d, d)
        self.Wa = nn.Linear(d, d, bias=False); self.va = nn.Linear(d, 1, bias=False); self.head = nn.Linear(d, V)
    def forward(self, src, tgt_in):
        H, hT = self.enc(self.emb(src)); s = hT[0]; outs = []; aligns = []
        keys = self.Wa(H)                                                     # [B, T, d]
        for t in range(tgt_in.shape[1]):
            e = self.emb(tgt_in[:, t])
            if self.att:
                score = self.va(torch.tanh(keys + s[:, None, :])).squeeze(-1)  # Bahdanau: v^T tanh(W_a h_j + U_a s)  (U_a folded into s for brevity)
                a = score.softmax(-1); ctx = (a[:, :, None] * H).sum(1); aligns.append(a)
                s = self.dec(torch.cat([e, ctx], -1), s)
            else:
                s = self.dec(e, s)
            outs.append(self.head(s))
        return torch.stack(outs, 1), (torch.stack(aligns, 1) if self.att else None)

def exp_seq2seq(quick):
    steps = 600 if quick else 2000; Ts = (8, 16) if quick else (8, 16, 32)
    print(f"\n== 4. seq2seq 'reverse the sequence': fixed-vector bottleneck vs attention, {steps} steps Adam 2e-3, batch 64 ==")
    V, d = 20, 64; torch.set_num_threads(THREADS)
    for T in Ts:
        row = []; last_align = None
        for att in (False, True):
            torch.manual_seed(0); m = Seq2Seq(V, d, att); opt = torch.optim.Adam(m.parameters(), 2e-3); g = torch.Generator().manual_seed(1); t = cli.Timer()
            for s in range(steps):
                src = torch.randint(0, V, (64, T), generator=g); tgt = src.flip(1)
                tgt_in = torch.cat([torch.full((64, 1), V), tgt[:, :-1]], 1)
                logits, _ = m(src, tgt_in); loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            with torch.no_grad():
                src = torch.randint(0, V, (1000, T), generator=g); tgt = src.flip(1)
                tgt_in = torch.cat([torch.full((1000, 1), V), tgt[:, :-1]], 1); logits, al = m(src, tgt_in)
                tok_acc = (logits.argmax(-1) == tgt).float().mean().item(); seq_acc = (logits.argmax(-1) == tgt).all(1).float().mean().item()
            row.append(f"{'attention' if att else 'bottleneck'}: token acc {tok_acc*100:5.1f}% seq acc {seq_acc*100:5.1f}% ({t})")
            if att: last_align = al[0]
        print(f"T={T:2d}: " + "   ".join(row))
        if T == 8:
            print("   alignment matrix (rows = output step, cols = input position), T=8, one example:")
            for r in last_align: print("   " + " ".join("#" if v > 0.5 else ("+" if v > 0.1 else ".") for v in r))

def exp_timing():
    print("\n== 5. forward wall time vs sequence length, d=256, batch 1, CPU: RNN (sequential) vs self-attention (parallel) ==")
    torch.set_num_threads(THREADS); d = 256
    rnn = nn.RNN(d, d, batch_first=True); attn = nn.MultiheadAttention(d, 4, batch_first=True)
    for T in (64, 256, 1024, 4096):
        x = torch.randn(1, T, d)
        with torch.no_grad():
            for _ in range(2): rnn(x); attn(x, x, x)
            t = cli.Timer(); [rnn(x) for _ in range(5)]; tr = t.secs() / 5
            t = cli.Timer(); [attn(x, x, x) for _ in range(5)]; ta = t.secs() / 5
        print(f"T={T:5d}: RNN {tr*1e3:8.1f} ms   self-attention {ta*1e3:8.1f} ms   (attention FLOPs ~ 4*T^2*d = {4*T*T*d/1e9:.2f} GFLOPs; RNN ~ 4*T*d^2 = {4*T*d*d/1e9:.3f} GFLOPs)")

if __name__ == "__main__":
    which, quick = cli.parse(EXPS, __doc__)
    if "bptt" in which: exp_bptt()
    if "memory" in which: exp_memory(quick)
    if "forget" in which: exp_forget(quick)
    if "seq2seq" in which: exp_seq2seq(quick)
    if "timing" in which: exp_timing()
