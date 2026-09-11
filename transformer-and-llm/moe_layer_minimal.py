"""一个最小 MoE 层（Transformer 与 LLM 05）：softmax 路由 + top-k + 逐专家循环 + 共享专家。只定义模块，供读者对照公式与调试。
https://arganzheng.life/moe-compute-and-communication.html
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d, d_ff, bias=False)
        self.w_up = nn.Linear(d, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoELayer(nn.Module):
    def __init__(self, d, d_ff, n_experts, top_k, n_shared=0):
        super().__init__()
        self.router = nn.Linear(d, n_experts, bias=False)       # W_r: d x E
        self.experts = nn.ModuleList(
            [SwiGLUExpert(d, d_ff) for _ in range(n_experts)])
        self.shared = nn.ModuleList(
            [SwiGLUExpert(d, d_ff) for _ in range(n_shared)])
        self.top_k = top_k

    def forward(self, x):                                       # x: [T, d]
        scores = F.softmax(self.router(x), dim=-1)              # [T, E]
        w, idx = scores.topk(self.top_k, dim=-1)                # [T, k]
        w = w / w.sum(dim=-1, keepdim=True)                     # 选中的 k 个重新归一化
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):               # 循环版：逐专家
            tok, slot = (idx == e).nonzero(as_tuple=True)       # 选了专家 e 的 token
            if tok.numel() == 0:
                continue
            out.index_add_(0, tok, w[tok, slot, None] * expert(x[tok]))
        for expert in self.shared:                              # 共享专家：所有 token
            out = out + expert(x)
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    layer = MoELayer(d=64, d_ff=128, n_experts=8, top_k=2, n_shared=1)
    x = torch.randn(16, 64)
    y = layer(x)
    print("out", tuple(y.shape), "params", sum(p.numel() for p in layer.parameters()))
    scores = F.softmax(layer.router(x), dim=-1)
    print("tokens per expert:", torch.bincount(scores.topk(2, dim=-1).indices.flatten(), minlength=8).tolist())
