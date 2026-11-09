"""反向传播 03 配套玩具与配图。所有数字由 CPU 实测或直接计算产生。

用法：python case_03_backprop_toys.py scalar|fanout|relu-vjp|jacobian|linear|softmax|mini-net|eps-sweep|autograd-graph|figures
"""
import importlib.util
import os
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
BLOG_IMG = "/Users/devin/repos/arganzheng-github-com/img/in-post"
EXPECTED = os.path.join(HERE, "expected")
spec = importlib.util.spec_from_file_location(
    "lst_plot", "/Users/devin/repos/ai-learning-labs/large-scale-training/_plot.py")
lst_plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lst_plot)
plt, save, text, box, box_title = (lst_plot.plt, lst_plot.save, lst_plot.text,
                                   lst_plot.box, lst_plot.box_title)
C = lst_plot.C
lst_plot.OUT = BLOG_IMG
torch.set_num_threads(1)


def _segment_hits_rect(a, b, rect):
    x0, y0, x1, y1 = rect
    dx, dy = b[0] - a[0], b[1] - a[1]
    lo, hi = 0.0, 1.0
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]),
                 (-dy, a[1] - y0), (dy, y1 - a[1])):
        if p == 0:
            if q <= 1e-4:
                return False
        else:
            r = q / p
            if p < 0:
                lo = max(lo, r)
            else:
                hi = min(hi, r)
            if lo >= hi:
                return False
    if lo >= hi:
        return False
    t = (lo + hi) / 2
    mid = (a[0] + t * dx, a[1] + t * dy)
    return x0 + 1e-4 < mid[0] < x1 - 1e-4 and y0 + 1e-4 < mid[1] < y1 - 1e-4


def assert_line_clear(ax, start, end):
    for patch in lst_plot._PATCHES:
        if patch.get_figure() is not ax.figure:
            continue
        x, y, w, h = patch.get_bbox().bounds
        assert not _segment_hits_rect(start, end, (x, y, x + w, y + h)), \
            f"line crosses box: {start} to {end}"


_SEGMENTS = []          # (ax, a, b) — arrow/dashed_line 画出的全部线段


def dashed_line(ax, start, end, color=C["orange"], lw=.9):
    assert_line_clear(ax, start, end)
    _SEGMENTS.append((ax, start, end))
    ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=lw, ls="--")


def arrow(ax, start, end, color="#555555", lw=1.2, style="-|>"):
    assert_line_clear(ax, start, end)
    _SEGMENTS.append((ax, start, end))
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style,
                                 mutation_scale=12, color=color, lw=lw,
                                 shrinkA=1, shrinkB=1))


def check_text_lines(fig):
    """文字框与已登记线段相交 = 失败。在 collide 的盒内检查之外补一层。"""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    errs = []
    for t, _, _ in lst_plot._TRACKED:
        if t.get_figure() is not fig or not t.get_text().strip():
            continue
        e = t.get_window_extent(renderer=ren)
        for ax, a, b in _SEGMENTS:
            if ax.figure is not fig:
                continue
            pa, pb = ax.transData.transform(a), ax.transData.transform(b)
            if _segment_hits_rect(pa, pb, (e.x0 + 1, e.y0 + 1, e.x1 - 1, e.y1 - 1)):
                errs.append(f"线段穿过文字 {t.get_text()!r}：{a} → {b}")
    if errs:
        raise AssertionError(f"{len(errs)} 处文字-线段相交：\n" + "\n".join(errs))


def finish(fig, name):
    check_text_lines(fig)
    save(fig, name)


def grid(ax, x, y, vals, cw=.5, ch=.5, fs=10.5, fc="#dce9f7", ec="#7a7a7a",
         zero_grey=True):
    """数字矩阵格子；y 为顶边，返回底边 y。"""
    for r, row in enumerate(vals):
        for c, v in enumerate(row):
            grey = zero_grey and v == 0
            ax.add_patch(Rectangle((x + c * cw, y - r * ch - ch), cw - .02,
                                   ch - .02, facecolor="#ececec" if grey else fc,
                                   edgecolor=ec, lw=.5))
            if not grey:
                ax.text(x + c * cw + (cw - .02) / 2, y - r * ch - ch / 2,
                        f"{v:g}", ha="center", va="center", fontsize=fs)
    return y - len(vals) * ch


def cell(ax, x, y, w, h, main, sub="", fc="#f5f8fc", ec="#777777", fs=10.5):
    """带数字的小格：主值 + 可选小字（未登记，不参与盒内检查）。"""
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=.8))
    ax.text(x + w / 2, y + h * .6 if sub else y + h / 2, main, ha="center",
            va="center", fontsize=fs)
    if sub:
        ax.text(x + w / 2, y + h * .22, sub, ha="center", va="center",
                fontsize=fs - 1.5, color=C["orange"])


def base_fig(title, w=7.6, h=4.3, xlim=(0, 12), ylim=(0, 7)):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    ax.set_title(title, fontweight=600, color="#111111", pad=10)
    return fig, ax


def node(ax, x, y, label, value="", grad="", fc="#f5f8fc", ec=C["blue"], w=1.45, h=1.1):
    p = box(ax, x, y, w, h, "", fc=fc, ec=ec, lw=1.0)
    box_title(ax, x+w/2, y+h*.73, label, p, color="#111111")
    if value:
        text(ax, x+w/2, y+h*.48, value, in_box=p, ha="center", va="center", fontsize=10.5, color=C["green"])
    if grad:
        text(ax, x+w/2, y+h*.2, grad, in_box=p, ha="center", va="center", fontsize=10.5, color=C["orange"])
    return p


def toy_scalar():
    x, w, b, y = 2.0, 3.0, -1.0, 4.0
    z=w*x+b; e=z-y; loss=e*e
    dz=2*e; dw=dz*x; db=dz; dx=dz*w
    eps=.001
    lp=((w+eps)*x+b-y)**2
    print("标量神经元：z=w*x+b, e=z-y, L=e^2")
    print(f"x={x:g}, w={w:g}, b={b:g}, y={y:g} -> z={z:g}, e={e:g}, L={loss:g}")
    print(f"dL/de={2*e:g}, dL/dz={dz:g}, dL/dw={dw:g}, dL/db={db:g}, dL/dx={dx:g}")
    print(f"前向有限差分 eps={eps:g}: w+eps={w+eps:.3f}, L(w+eps)={lp:.6f}, dL/dw≈{(lp-loss)/eps:.6f}")


def toy_fanout():
    x,y=2.,3.
    print("f=x*y+x; x=2, y=3")
    print(f"f={x*y+x:g}; 路径 x*y 的梯度贡献 y={y:g}; 路径 x 的贡献 1; 汇合后 df/dx={y+1:g}")
    print("若 batch 中每个样本都使用 W，则 dW 是各样本贡献之和；PyTorch .grad 也按调用累加。")


def toy_relu():
    x=torch.tensor([-1.,2.,.5],dtype=torch.float64,requires_grad=True)
    y=torch.relu(x); gy=torch.tensor([.3,-.2,.5],dtype=torch.float64)
    gx=torch.autograd.grad(y,x,gy)[0]
    print("x=(-1, 2, 0.5); y=ReLU(x)")
    print("J=diag(0,1,1)")
    print(f"g_y={tuple(gy.tolist())}; g_x=J^T*g_y={tuple(gx.tolist())}")
    print("等价计算：g_x = g_y 逐元素乘 (0,1,1)")


def toy_jacobian():
    m,k,n=3,2,2
    X=torch.tensor([[1.,2.],[3.,4.],[5.,6.]],dtype=torch.float64)
    W=torch.tensor([[1.,2.],[-1.,.5]],dtype=torch.float64,requires_grad=True)
    def f(w): return X @ w
    J=torch.autograd.functional.jacobian(f,W)
    # Jacobian of flattened output to flattened input for one sample's linear map
    # The requested sample-coupling structure is captured by X_batch @ W only
    # when differentiating outputs with respect to batch inputs; construct that map below.
    xb=torch.arange(1.,m*k+1,dtype=torch.float64).reshape(m,k).requires_grad_()
    yb=xb @ torch.eye(k,n,dtype=torch.float64)
    Jx=torch.autograd.functional.jacobian(lambda q: q @ W.detach(), xb)
    Jflat=Jx.reshape(m*n,m*k)
    print(f"Linear batch: m={m}, k={k}, n={n}; W={W.detach().tolist()}")
    print("对 batch 输入求导得到的 Jacobian（输出展平 × 输入展平）:")
    print(Jflat.numpy())
    expected=np.kron(np.eye(m),W.detach().numpy().T)
    assert np.allclose(Jflat.numpy(),expected)
    print(f"shape={tuple(Jflat.shape)}；非对角样本块全为 0；每个对角块都是 W^T。")
    print("对单样本 y=xW，J=W^T；VJP J^T*g = W*g。")
    print(f"用于 F6 尺寸演示：m*n={m*n}, m*k={m*k}, Jacobian 元素={(m*n)*(m*k)}。")


def toy_linear():
    X=np.array([[1.,2.,-1.],[.5,-1.,2.]])
    W=np.array([[1.,-.5],[.25,2.],[-1.,.75]])
    G=np.array([[.2,-.3],[1.,.5]])
    dW=X.T@G; dX=G@W.T; db=G.sum(0)
    print("Linear 数字例：m=2,k=3,n=2")
    print(f"X=\n{X}\nW=\n{W}\nG=\n{G}")
    print(f"dW=X^T G=\n{dW}\ndX=G W^T=\n{dX}\ndb=sum_rows(G)={db}")
    print(f"逐样本外积之和误差={np.max(np.abs(dW-(np.outer(X[0],G[0])+np.outer(X[1],G[1])))):.1e}")


def toy_softmax():
    z=np.array([2.,1.,.1]); p=np.exp(z-z.max()); p/=p.sum(); y=np.array([1.,0.,0.])
    print("softmax + cross entropy：z=(2,1,0.1)，正确类=0")
    print("p="+np.array2string(p,precision=3))
    print("dL/dz=p-y="+np.array2string(p-y,precision=3))


def mini_values():
    x=np.array([.6,-1.2]); W1=np.array([[.5,-.3],[.8,.4]]); b1=np.array([1.0,0.0])
    W2=np.array([[.7,-.5],[.2,.6]]); b2=np.array([.05,-.1]); target=0
    h=x@W1+b1; a=np.maximum(h,0); z=a@W2+b2
    p=np.exp(z-z.max()); p/=p.sum(); loss=-np.log(p[target])
    gz=p.copy(); gz[target]-=1
    dw2=np.outer(a,gz); db2=gz; ga=W2@gz; gh=ga*(h>0); dw1=np.outer(x,gh); db1=gh
    return locals()


def mini_rel_err():
    d=mini_values()
    def loss_for(w1,w2):
        h=d["x"]@w1+d["b1"]; a=np.maximum(h,0); z=a@w2+d["b2"]
        p=np.exp(z-z.max()); p/=p.sum()
        return -np.log(p[d["target"]])
    eps=1e-6; errs=[]
    for name in ("W1","W2"):
        a=d[name]; g=d["dw1" if name=="W1" else "dw2"]
        for ix in np.ndindex(a.shape):
            old=a[ix]; a[ix]=old+eps; lp=loss_for(d["W1"],d["W2"]); a[ix]=old-eps; lm=loss_for(d["W1"],d["W2"]); a[ix]=old
            num=(lp-lm)/(2*eps); errs.append(abs(num-g[ix])/(abs(num)+abs(g[ix])+1e-15))
    return max(errs)


def toy_mini():
    d=mini_values()
    for key in ("x","W1","b1","h","a","W2","b2","z","p"):
        print(f"{key}={np.array2string(d[key],precision=6)}")
    print(f"target={d['target']}; loss={d['loss']:.9f}")
    for key in ("gz","dw2","db2","ga","gh","dw1","db1"):
        print(f"{key}={np.array2string(d[key],precision=6)}")
    print(f"中心差分 eps=1e-6：dW1、dW2 最大相对误差={mini_rel_err():.3e}")


def toy_eps():
    x0=.7
    def f(x): return np.sin(x)*np.exp(x)
    exact=np.exp(x0)*(np.sin(x0)+np.cos(x0))
    epses=np.logspace(-1,-10,10)
    print(f"f(x)=exp(x)sin(x), x={x0}; exact derivative={exact:.12g}")
    print("dtype eps relative_error")
    results={}
    for dtype in (np.float32,np.float64):
        vals=[]
        for e in epses:
            xx=dtype(x0); ee=dtype(e)
            fp=dtype(np.exp(dtype(xx+ee))*np.sin(dtype(xx+ee)))
            fm=dtype(np.exp(dtype(xx-ee))*np.sin(dtype(xx-ee)))
            num=dtype((fp-fm)/dtype(2*ee))
            rel=abs(float(num)-exact)/(abs(float(num))+abs(exact))
            vals.append(rel); print(f"{np.dtype(dtype).name} {e:.0e} {rel:.6e}")
        results[dtype]=np.array(vals)
    return epses,results


def graph_nodes():
    torch.manual_seed(17)
    model=nn.Sequential(nn.Linear(3,4),nn.ReLU(),nn.Linear(4,2))
    x=torch.tensor([[.2,-.4,.7]])
    y=torch.tensor([1])
    loss=F.cross_entropy(model(x),y)
    pname={p.data_ptr():f"层{i//2+1} {'W' if n.endswith('weight') else 'b'}"
           for i,(n,p) in enumerate(model.named_parameters())}
    root=loss.grad_fn
    nodes=[]; index={}; q=deque([root])
    while q:
        node=q.popleft()
        if node is None or node in index: continue
        index[node]=len(nodes); nodes.append(node)
        for nxt,_ in node.next_functions:
            if nxt is not None and nxt not in index: q.append(nxt)
    edges=[]
    for node in nodes:
        for nxt,_ in node.next_functions:
            if nxt is not None: edges.append((index[node],index[nxt]))
    leaves=[pname.get(getattr(n,'variable',None).data_ptr()
                      if getattr(n,'variable',None) is not None else None)
            for n in nodes]
    return loss.item(),[node.name() for node in nodes],edges,leaves


def toy_graph():
    loss,order,edges,leaves=graph_nodes()
    print(f"loss={loss:.9f}; BFS 遍历 loss.grad_fn.next_functions：")
    for i,name in enumerate(order):
        print(f"{i:02d} {name}"+(f"   <- {leaves[i]}" if leaves[i] else ""))
    print("edges="+", ".join(f"{a}->{b}" for a,b in edges))
    print(f"unique autograd nodes={len(order)}")


def figure_overview():
    fig,ax=base_fig("前向算 loss，反向沿原路把梯度送回每个参数",7.6,3.9,(0,12),(1.3,7.0))
    xs=[.35,2.55,4.75,6.95,9.15]; y=4.7; w=1.6
    fw=[("X","输入"),("H","Linear"),("A","ReLU"),("Z","Linear"),("L","loss")]
    for x,(a,b) in zip(xs,fw):
        p=box(ax,x,y,w,1.0,"",fc="#dce9f7",ec=C["blue"])
        box_title(ax,x+w/2,y+.68,a,p); text(ax,x+w/2,y+.27,b,in_box=p,ha="center",va="center",fontsize=10.5)
    for i in range(4): arrow(ax,(xs[i]+w+.05,y+.5),(xs[i+1]-.08,y+.5),color=C["blue"])
    text(ax,.4,6.15,"前向：逐层计算",ha="left",va="center",fontsize=10.5,color=C["blue"])
    for x,lab in ((2.75,"W1 参数"),(7.15,"W2 参数")):
        p=box(ax,x,6.0,1.5,.65,"",fc="#f0f0f0",ec="#888888")
        text(ax,x+.75,6.32,lab,in_box=p,ha="center",va="center",fontsize=10.5)
        arrow(ax,(x+.75,5.97),(x+.75,5.75),color="#888888",lw=.8)
    band=3.9
    text(ax,10.85,band,"反向：逐节点回走",ha="left",va="center",fontsize=10.5,color=C["orange"])
    cxs=[x+w/2 for x in xs]
    for i in (4,3,2,1):
        arrow(ax,(cxs[i]-.15,band),(cxs[i-1]+.15,band),color=C["orange"])
    labs=[r"$G_X$（不算）",r"$\times$mask $=G_H$",r"$\times W_2^{\!\!\top}=G_A$",r"$G_Z=(P-Y)/m$"]
    for i,(a,b) in enumerate(((0,1),(1,2),(2,3),(3,4))):
        mid=(cxs[a]+cxs[b])/2
        text(ax,mid,band-.42,labs[i],ha="center",va="center",fontsize=10.5,color=C["orange"])
    for cx,lab in ((cxs[1],r"$dW_1=X^{\!\!\top}G_H$"),(cxs[3],r"$dW_2=A^{\!\!\top}G_Z$")):
        p=box(ax,cx-1.3,1.85,2.6,.85,"",fc="#fff3e0",ec=C["orange"])
        box_title(ax,cx,2.28,lab,p,color="#111111")
        arrow(ax,(cx,band-.05),(cx,2.75),color=C["orange"])
    finish(fig,"backprop-overview")


def figure_scalar():
    fig,ax=base_fig("标量计算图：每条边反向下传「上游梯度 × 局部导数」",7.6,3.8,(0,12),(0.9,6.6))
    top=4.9
    node(ax,.7,top,r"$x$","2",w=1.4,h=1.1)
    node(ax,3.7,top,r"$z=wx+b$","5",w=2.4,h=1.1)
    node(ax,7.1,top,"e=z-y","1",w=2.0,h=1.1)
    node(ax,10.1,top,r"$L=e^2$","1",w=1.5,h=1.1)
    node(ax,2.6,1.3,r"$w$","3",w=1.5,h=1.05)
    node(ax,5.1,1.3,r"$b$","-1",w=1.5,h=1.05)
    node(ax,8.2,1.3,r"$y$","4",w=1.5,h=1.05)
    arrow(ax,(2.15,5.45),(3.65,5.45),color=C["blue"])
    arrow(ax,(6.15,5.45),(7.05,5.45),color=C["blue"])
    arrow(ax,(9.15,5.45),(10.05,5.45),color=C["blue"])
    arrow(ax,(3.35,2.4),(4.5,4.85),color=C["blue"])
    arrow(ax,(5.85,2.4),(5.3,4.85),color=C["blue"])
    arrow(ax,(8.95,2.4),(8.35,4.85),color=C["blue"])
    arrow(ax,(3.6,5.1),(2.2,5.1),color=C["orange"],lw=.9)
    arrow(ax,(7.0,5.1),(6.2,5.1),color=C["orange"],lw=.9)
    arrow(ax,(10.0,5.1),(9.2,5.1),color=C["orange"],lw=.9)
    arrow(ax,(4.55,4.8),(3.5,2.45),color=C["orange"],lw=.9)
    arrow(ax,(5.15,4.8),(5.7,2.45),color=C["orange"],lw=.9)
    arrow(ax,(8.45,4.8),(9.05,2.45),color=C["orange"],lw=.9)
    lab=[((2.9,4.5),(r"$\times w=3$" + "\n" + "$dL/dx=6$"),"center"),((6.6,4.5),(r"$\times1$" + "\n" + "$dL/dz=2$"),"center"),
         ((9.6,4.5),(r"$\times2e=2$" + "\n" + "$dL/de=2$"),"center"),((2.45,3.05),(r"$\times x=2$" + "\n" + "$dL/dw=4$"),"right"),
         ((6.15,3.35),(r"$\times1$" + "\n" + "$dL/db=2$"),"left"),((8.9,3.67),(r"$\times{-1}$" + "\n" + "$dL/dy=-2$"),"left")]
    for (x,y),s,ha in lab:
        text(ax,x,y,s,ha=ha,va="center",fontsize=10.5,color=C["orange"])
    finish(fig,"backprop-scalar-graph")


def figure_gate():
    """三种局部反向规则：加法分发、乘法换路、max 只走被选中的一路。

    每列：蓝色 = 前向（输入值 → 输出值），橙色 = 反向（上游梯度 g=4
    从输出端进来，乘上这条边的局部导数，得到该输入的下游梯度）。
    与 figure_scalar 同一套配色，边旁标出「× 局部导数」与结果。
    """
    fig,ax=base_fig("每个节点只做一件事：下游梯度 = 上游梯度 × 局部导数",
                    7.6,4.8,(0,12),(0.2,9.2))
    # 图例（顶部一行）
    ax.plot([0.6,1.4],[8.95,8.95],color=C["blue"],lw=1.8)
    text(ax,1.6,8.95,"前向（数值）",ha="left",va="center",fontsize=10.5,color=C["blue"])
    ax.plot([4.0,4.8],[8.95,8.95],color=C["orange"],lw=1.8)
    text(ax,5.0,8.95,"反向（梯度）",ha="left",va="center",fontsize=10.5,color=C["orange"])
    text(ax,9.3,8.95,r"上游梯度 $g=4$ 相同，分配规则不同",
         ha="left",va="center",fontsize=10.5,color="#555555")
    # 三行横排；每行自含：行题在最左，输入 → gate → 输出，g 从最右进来
    for y in (5.85,3.15):
        ax.plot([0.5,11.6],[y,y],color="#d9dee6",lw=1)
    # (行题, 算符, 局部导数 u', v', 输出值, du, dv, du算式, dv算式, 说明)
    gates=[("加法门",r"$o=u+v$","1","1","5",r"$du=4\times1=4$",r"$dv=4\times1=4$","原样分发"),
           ("乘法门",r"$o=u\,v$","v=3","u=2","6",r"$du=4\times3=12$",r"$dv=4\times2=8$","乘上另一路输入"),
           ("max 门",r"$o=\max(u,v)$","0","1","3",r"$du=4\times0=0$",r"$dv=4\times1=4$","只给被选中的一路")]
    for cy,(op,fo,du_l,dv_l,o,g1,g2,rule) in zip((7.15,4.5,1.85),gates):
        text(ax,.55,cy,op,ha="left",va="center",fontsize=11)
        # 输入 u=2 (上) v=3 (下)
        cell(ax,1.9,cy+.45,1.15,.75,r"$u=2$",fc="#dce9f7",ec=C["blue"])
        cell(ax,1.9,cy-1.3,1.15,.75,r"$v=3$",fc="#dce9f7",ec=C["blue"])
        p=box(ax,3.85,cy-.6,1.9,1.2,"",fc="#f5f8fc",ec="#777777")
        text(ax,4.8,cy,fo,in_box=p,ha="center",va="center",fontsize=10.5)
        cell(ax,6.05,cy-.4,1.0,.8,r"$o="+o+"$",fc="#f5f8fc",ec=C["blue"])
        # 前向箭头（蓝，上侧）与反向箭头（橙，下侧）平行错开
        arrow(ax,(3.1,cy+.8),(3.8,cy+.4),color=C["blue"],lw=1.1)
        arrow(ax,(3.8,cy+.15),(3.1,cy+.55),color=C["orange"],lw=1.0)
        arrow(ax,(3.1,cy-.95),(3.8,cy-.5),color=C["blue"],lw=1.1)
        arrow(ax,(3.8,cy-.85),(3.1,cy-1.2),color=C["orange"],lw=1.0)
        arrow(ax,(5.8,cy+.15),(6.0,cy+.15),color=C["blue"],lw=1.1)
        arrow(ax,(6.0,cy-.18),(5.8,cy-.18),color=C["orange"],lw=1.0)
        # 上游梯度 g=4 从右侧进来（橙），标注就压在箭头旁
        arrow(ax,(8.35,cy),(7.1,cy),color=C["orange"],lw=1.1)
        text(ax,7.72,cy+.35,r"上游 $g=4$",ha="center",fontsize=10.5,color=C["orange"])
        # 反向结果：算式+结果合并写在输入格外侧
        text(ax,.62,cy+.75,g1,ha="center",fontsize=10.5,color=C["orange"])
        text(ax,.62,cy-.85,g2,ha="center",fontsize=10.5,color=C["orange"])
        # 行尾：局部导数与规则
        text(ax,9.1,cy,r"$\partial o/\partial u="+du_l+r"$",ha="left",va="center",fontsize=10.5)
        text(ax,9.1,cy-.5,r"$\partial o/\partial v="+dv_l+r"$",ha="left",va="center",fontsize=10.5)
        text(ax,9.1,cy-1.05,rule,ha="left",va="center",fontsize=10.5)
    finish(fig,"backprop-local-vjp")


def figure_fanout():
    fig,ax=base_fig("一个量走了两条路：两路梯度在 x 处相加",7.6,3.7,(0,12),(0.4,6.4))
    node(ax,.5,3.75,r"$x$","2",w=1.3,h=1.4)
    node(ax,.5,1.15,r"$y$","3",w=1.3,h=1.4)
    node(ax,3.4,3.65,"x*y","6",w=1.8,h=1.45)
    node(ax,6.6,2.35,"+","f=8",w=1.6,h=1.35)
    arrow(ax,(1.85,4.3),(3.35,4.3),color=C["blue"])
    arrow(ax,(1.85,1.9),(3.35,3.8),color=C["blue"])
    arrow(ax,(1.15,3.7),(1.15,2.7),color=C["blue"])
    arrow(ax,(1.15,2.7),(6.5,2.7),color=C["blue"])
    arrow(ax,(5.25,3.9),(6.55,3.2),color=C["blue"])
    arrow(ax,(9.3,3.0),(8.35,3.0),color=C["orange"]); text(ax,9.45,3.35,"上游 g=1",ha="left",fontsize=10.5,color=C["orange"])
    arrow(ax,(6.5,3.35),(5.35,4.15),color=C["orange"],lw=.9)
    text(ax,6.35,4.0,"1",ha="center",fontsize=10.5,color=C["orange"])
    arrow(ax,(6.55,2.7),(1.45,2.7),color=C["orange"],lw=.9)
    arrow(ax,(1.25,2.6),(1.25,3.7),color=C["orange"],lw=.9)
    text(ax,4.0,2.15,"1",ha="center",fontsize=10.5,color=C["orange"])
    arrow(ax,(3.3,4.55),(1.9,4.55),color=C["orange"],lw=.9)
    text(ax,2.6,5.5,r"$\times y=3$，$g=3$",ha="center",fontsize=10.5,color=C["orange"])
    arrow(ax,(3.3,3.6),(1.95,1.6),color=C["orange"],lw=.9)
    text(ax,3.15,2.85,r"$\times x=2$，$g=2$",ha="left",fontsize=10.5,color=C["orange"])
    text(ax,0.3,3.45,"+",ha="center",va="center",fontsize=13,color=C["orange"])
    text(ax,0.6,3.05,"3 + 1 = 4",ha="center",va="center",fontsize=10.5,color=C["orange"])
    text(ax,6,0.85,"加法分叉两路各得 1；乘法门把 1 乘上对方的值；同一输入的两路梯度相加。",ha="center",fontsize=10.5)
    finish(fig,"backprop-fanout")


def figure_shapes():
    fig,ax=base_fig("梯度与被求导的量同形；向量对向量才得到 Jacobian",7.6,3.6,(0,12),(0.6,6.9))
    text(ax,2.2,5.55,r"标量 $L$ 对向量 $x$",ha="center",fontsize=10.5)
    grid(ax,.6,4.95,[[1,2,3]],cw=.5,zero_grey=False)
    arrow(ax,(2.25,4.4),(2.75,4.4),color=C["blue"])
    grid(ax,2.85,4.95,[[2,4,6]],cw=.5,fc="#fae6d6",zero_grey=False)
    text(ax,2.35,3.45,(r"$L=\sum_i x_i^2$" + "\n" + "$dL/dx$ 与 $x$ 同形"),ha="center",va="center",fontsize=10.5)
    text(ax,5.95,5.55,r"标量 $L$ 对矩阵 $W$",ha="center",fontsize=10.5)
    grid(ax,4.35,4.95,[[1,2],[3,4]],cw=.5,zero_grey=False)
    arrow(ax,(5.5,4.4),(6.0,4.4),color=C["blue"])
    grid(ax,6.1,4.95,[[2,4],[6,8]],cw=.5,fc="#fae6d6",zero_grey=False)
    text(ax,5.95,3.45,(r"$L=\sum_{ij}W_{ij}^2$" + "\n" + "$dW$ 与 $W$ 同形"),ha="center",va="center",fontsize=10.5)
    text(ax,9.5,5.55,r"向量 $y$ 对向量 $x$：Jacobian",ha="center",fontsize=10.5)
    grid(ax,8.1,4.95,[[1,1,0],[0,3,2]],cw=.5,zero_grey=False)
    text(ax,9.85,4.42,r"$y_1$ 的梯度",ha="left",va="center",fontsize=10.5)
    text(ax,9.85,3.92,r"$y_2$ 的梯度",ha="left",va="center",fontsize=10.5)
    text(ax,11.7,3.3,(r"$y=(x_1+x_2,\,x_2 x_3)$" + "\n" + r"$x=(1,2,3)$，$J$ 为 $2\times3$"),ha="right",va="center",fontsize=10.5)
    text(ax,6,2.35,r"每格 $g_{ij}$：$W_{ij}$ 增 $\epsilon$，$L$ 约增 $g_{ij}\epsilon$。$W\leftarrow W-\eta\,dW$ 逐元素对应，",ha="center",fontsize=10.5)
    text(ax,6,1.9,"所以标量对参数的梯度必须与参数同形；只有向量对向量才产生 Jacobian。",ha="center",fontsize=10.5)
    finish(fig,"backprop-gradient-shapes")


def figure_relu_jacobian():
    fig,ax=base_fig("ReLU 的 Jacobian 是对角掩码：VJP 等于逐元素乘",7.6,3.3,(0,12),(0.5,6.6))
    text(ax,.5,6.0,r"$x=(-1,\,2,\,0.5)$；mask$_i=1$ 当 $x_i>0$，否则 $0$",ha="left",fontsize=10.5)
    text(ax,1.25,5.15,r"$J$",ha="center",fontsize=10.5)
    grid(ax,.5,4.95,[[0,0,0],[0,1,0],[0,0,1]],cw=.5)
    text(ax,3.0,5.15,r"$g_y$",ha="center",fontsize=10.5)
    grid(ax,2.75,4.95,[[.3],[-.2],[.5]],cw=.5,zero_grey=False)
    text(ax,4.45,5.15,r"$g_x$",ha="center",fontsize=10.5)
    grid(ax,4.2,4.95,[[0],[-.2],[.5]],cw=.5,fc="#fae6d6",zero_grey=False)
    text(ax,2.35,4.15,r"$\times$",ha="center",va="center",fontsize=12)
    text(ax,3.85,4.15,"=",ha="center",va="center",fontsize=11)
    text(ax,5.7,4.15,r"$g_x=J^{\!\!\top}g_y$；负元素的梯度被挡住",ha="left",va="center",fontsize=10.5)
    text(ax,1.25,2.75,"mask",ha="center",fontsize=10.5)
    grid(ax,.5,2.55,[[0,1,1]],cw=.5,zero_grey=False)
    text(ax,3.5,2.75,r"$g_y$",ha="center",fontsize=10.5)
    grid(ax,2.9,2.55,[[.3,-.2,.5]],cw=.5,zero_grey=False)
    text(ax,5.45,2.75,r"$g_x$",ha="center",fontsize=10.5)
    grid(ax,4.85,2.55,[[0,-.2,.5]],cw=.5,fc="#fae6d6",zero_grey=False)
    text(ax,2.2,2.15,r"$\times$",ha="center",va="center",fontsize=12)
    text(ax,4.6,2.15,"=",ha="center",va="center",fontsize=11)
    text(ax,6.85,2.15,r"逐元素乘：同一结果，不建 $3\times3$",ha="left",va="center",fontsize=10.5)
    finish(fig,"backprop-relu-jacobian")


def figure_batched_jacobian():
    J=np.array([[1.,-1.,0,0,0,0],[2.,.5,0,0,0,0],
                [0,0,1.,-1.,0,0],[0,0,2.,.5,0,0],
                [0,0,0,0,1.,-1.],[0,0,0,0,2.,.5]])
    W=np.array([[1.,2.],[-1.,.5]]); G=np.array([[.2,-.3],[1.,.5],[.7,-.1]])
    dX=G@W.T
    assert np.allclose(dX.reshape(-1),J.T@G.reshape(-1))
    fig,ax=base_fig("batch Linear 的 Jacobian 分块对角：VJP 就是矩阵乘",7.6,3.8,(0,12),(0.5,6.2))
    text(ax,1.5,5.75,r"$6\times6$ Jacobian（toy 实测）",ha="center",fontsize=10.5)
    cell=.37; x0,y0=.4,5.3
    for r in range(6):
        for c in range(6):
            v=J[r,c]; grey=v==0
            ax.add_patch(Rectangle((x0+c*cell,y0-(r+1)*cell),cell-.015,cell-.015,
                                   facecolor="#ececec" if grey else "#dce9f7",
                                   edgecolor="#888888",lw=.4))
            if not grey:
                ax.text(x0+c*cell+cell/2,y0-(r+.5)*cell,f"{v:g}",ha="center",va="center",fontsize=10.5)
    for bx,by in ((x0,y0),(x0+2*cell,y0-2*cell),(x0+4*cell,y0-4*cell)):
        ax.add_patch(Rectangle((bx,by-2*cell),2*cell,2*cell,fill=False,edgecolor="#111111",lw=1.4))
    text(ax,1.5,2.75,(r"对角块都是 $W^{\!\!\top}$，其余全 $0$" + "\n" + "（样本互不依赖）"),ha="center",va="center",fontsize=10.5)
    text(ax,5.2,5.15,r"$G$（$3\times2$）",ha="center",fontsize=10.5)
    grid(ax,4.7,4.9,G.tolist(),cw=.5,zero_grey=False,fs=10.5)
    text(ax,6.65,4.15,"×",ha="center",va="center",fontsize=12)
    text(ax,7.75,5.15,r"$W^{\!\!\top}$",ha="center",fontsize=10.5)
    grid(ax,7.35,4.9,W.T.tolist(),cw=.5,zero_grey=False,fs=10.5)
    text(ax,8.7,4.15,"=",ha="center",va="center",fontsize=11)
    text(ax,10.0,5.15,r"$dX$（$3\times2$）",ha="center",fontsize=10.5)
    grid(ax,9.4,4.9,dX.tolist(),cw=.5,fc="#fae6d6",zero_grey=False,fs=10.5)
    text(ax,7.7,3.1,r"乘大矩阵 = 直接 $G\,W^{\!\!\top}$，结果一致",ha="center",va="center",fontsize=10.5)
    text(ax,4.2,2.0,"本文第一层 batch 128：",ha="left",fontsize=10.5)
    text(ax,4.2,1.6,r"$(128\times784)\times(128\times256)\times4$B",ha="left",fontsize=10.5)
    vals=[("Jacobian",2.3,"13 GB",C["red"]),(r"$dX$",.55,"392 KiB",C["blue"]),(r"$W_1$",.7,"784 KiB",C["green"])]
    for i,(lab,wdt,value,c) in enumerate(vals):
        yy=2.15-i*.45
        text(ax,8.35,yy,lab,ha="right",va="center",fontsize=10.5)
        ax.add_patch(Rectangle((8.55,yy-.11),wdt,.2,facecolor=c,edgecolor="none"))
        text(ax,11.9,yy,value,ha="right",va="center",fontsize=10.5)
    finish(fig,"backprop-no-full-jacobian")


def figure_linear_outer():
    X=np.array([[1.,2.,-1.],[.5,-1.,2.]]); G=np.array([[.2,-.3],[1.,.5]])
    O1=np.outer(X[0],G[0]); O2=np.outer(X[1],G[1]); dW=X.T@G
    fig,ax=base_fig("Linear 权重梯度：每个样本贡献一个外积",7.6,3.3,(0,12),(0.8,6.5))
    text(ax,1.2,5.45,r"样本1 外积 $x_1^{\!\!\top}g_1$",ha="center",fontsize=10.5)
    grid(ax,.5,5.0,O1.tolist(),cw=.62,fs=10.5)
    text(ax,1.95,4.2,"+",ha="center",va="center",fontsize=12)
    text(ax,3.2,5.45,r"样本2 外积 $x_2^{\!\!\top}g_2$",ha="center",fontsize=10.5)
    grid(ax,2.5,5.0,O2.tolist(),cw=.62,fs=10.5)
    text(ax,3.95,4.2,"=",ha="center",va="center",fontsize=11)
    text(ax,4.95,5.45,r"$dW$",ha="center",fontsize=10.5)
    grid(ax,4.35,5.0,dW.tolist(),cw=.62,fc="#fae6d6",fs=10.5)
    text(ax,6.3,4.2,"与 W 同形",ha="left",va="center",fontsize=10.5)
    text(ax,.5,3.1,r"矩阵乘 $X^{\!\!\top}G$ 一次算出的正是这份和：",ha="left",fontsize=10.5)
    text(ax,.5,2.6,r"$dW[r,j]=\sum_i x_i[r]\,g_i[j]$ —— 每个样本 $i$ 的外积取 $(r,j)$ 项相加。",ha="left",fontsize=10.5)
    text(ax,.5,2.0,"batch 维求和 = 二.3 的分叉相加：同一 W 被每个样本各用一次。",ha="left",fontsize=10.5)
    finish(fig,"backprop-linear-outer-products")


def figure_softmax():
    z=np.array([2.,1.,.1]); p=np.exp(z-z.max()); p/=p.sum(); grad=p-np.array([1.,0.,0.])
    fig,ax=base_fig("softmax + 交叉熵：正确类上推，其余类按概率下压",7.6,3.0,(0,12),(0.4,4.9))
    xs=[3,6,9]
    for i,x in enumerate(xs):
        h=.5+abs(grad[i])*2.6; y=1.55
        c=C["green"] if i==0 else C["orange"]
        ax.add_patch(Rectangle((x-.52,y),1.04,h,facecolor=c,alpha=.28,edgecolor=c,lw=1.2))
        ax.text(x,y+h+.18,f"z={z[i]:g}",ha="center",va="bottom",fontsize=10.5)
        ax.text(x,y+.15,f"p={p[i]:.3f}\ng={grad[i]:+.3f}",ha="center",va="bottom",fontsize=10.5)
    text(ax,3,4.35,"正确类",ha="center",fontsize=10.5,color=C["green"])
    text(ax,6,.85,r"$dL/dz=p-\mathrm{onehot}(y)$：正确类得到负梯度上推，其余按概率下压。",ha="center",fontsize=10.5)
    finish(fig,"backprop-softmax-ce")


def figure_mini():
    d=mini_values(); err=mini_rel_err()
    fig,ax=base_fig("2-2-2 迷你网络：前向值在上，反向梯度沿链逐节点回走",7.6,3.8,(0,12),(0.3,6.3))
    xs=[.3,2.75,5.2,7.65,10.1]; w=1.8
    vals=[("x","[ 0.6 -1.2]"),("h","[0.34 -0.66]"),("a","[0.34 0]"),
          ("z","[0.29 -0.27]"),("L","0.453")]
    for x,(a,b) in zip(xs,vals):
        p=box(ax,x,4.5,w,1.05,"",fc="#dce9f7",ec=C["blue"])
        box_title(ax,x+w/2,5.2,a,p); text(ax,x+w/2,4.78,b,in_box=p,ha="center",va="center",fontsize=10.5)
    for i in range(4): arrow(ax,(xs[i]+w+.05,5.02),(xs[i+1]-.08,5.02),color=C["blue"])
    text(ax,4.88,4.05,"ReLU 置 0",ha="center",va="center",fontsize=10.5)
    bw=1.9
    grads=[(2.7,r"$G_H$","[-0.44 0]"),(5.15,r"$G_A$","[-0.44 0.15]"),(7.6,r"$G_Z$","[-0.36 0.36]")]
    for x,a,b in grads:
        p=box(ax,x,2.6,bw,1.0,"",fc="#fff3e0",ec=C["orange"])
        box_title(ax,x+bw/2,3.28,a,p); text(ax,x+bw/2,2.86,b,in_box=p,ha="center",va="center",fontsize=10.5)
    arrow(ax,(8.55,4.45),(8.55,3.65),color=C["orange"])
    arrow(ax,(6.1,4.45),(6.1,3.65),color=C["orange"])
    arrow(ax,(3.65,4.45),(3.65,3.65),color=C["orange"])
    arrow(ax,(7.55,3.1),(7.15,3.1),color=C["orange"])
    arrow(ax,(5.1,3.1),(4.7,3.1),color=C["orange"])
    text(ax,10.35,3.4,r"$G_Z=(P-Y)$",ha="left",va="center",fontsize=10.5,color=C["orange"])
    for x,a,b in ((1.4,"dW1","[[-0.26 0]\n [ 0.52 0]]"),(5.6,"dW2","[[-0.12 0.12]\n [0 0]]")):
        p=box(ax,x,.55,2.5,1.55,"",fc="#fff3e0",ec=C["orange"])
        box_title(ax,x+1.25,1.78,a,p); text(ax,x+1.25,1.1,b,in_box=p,ha="center",va="center",fontsize=10.5)
    arrow(ax,(3.7,2.55),(3.0,2.15),color=C["orange"],lw=.9)
    arrow(ax,(8.35,2.55),(7.85,2.15),color=C["orange"],lw=.9)
    text(ax,4.5,.3,f"有限差分最大相对误差 = {err:.1e}",ha="center",fontsize=10.5)
    p=box(ax,8.35,.30,3.55,1.95,"",fc="#f5f5f5",ec="#888888")
    text(ax,10.125,1.28,"参数 W1=[[.5,-.3],[.8,.4]]\n  b1=[1,0]\n  W2=[[.7,-.5],[.2,.6]]\n  b2=[.05,-.1]",in_box=p,ha="center",va="center",fontsize=10.5)
    finish(fig,"backprop-mini-network")


def figure_mlp():
    fig,ax=base_fig("784-256-10：前向、主梯度链、参数梯度分三排",7.6,4.2,(0,12),(0.6,7))
    xs=[.45,2.75,5.05,7.35,9.65]
    fw=[(r"$X$",r"$128\times784$"),(r"$H$",r"$128\times256$"),(r"$A$",r"$128\times256$"),(r"$Z$",r"$128\times10$"),(r"$L$","scalar")]
    for x,(a,b) in zip(xs,fw):
        p=box(ax,x,5.22,1.55,1.05,"",fc="#dce9f7",ec=C["blue"]); box_title(ax,x+.775,5.96,a,p)
        text(ax,x+.775,5.5,b,in_box=p,ha="center",va="center",fontsize=10.5)
    for i in range(4): arrow(ax,(xs[i]+1.6,5.72),(xs[i+1]-.05,5.72),color=C["blue"])
    tags=[(.45,r"存 $X$"),(2.75,r"存 $H$ 符号"),(5.05,r"存 $A$")]
    for x,label in tags:
        p=box(ax,x,4.42,1.55,.52,"",fc="#fae6d6",ec=C["orange"],lw=.8)
        text(ax,x+.775,4.68,label,in_box=p,ha="center",va="center",fontsize=10.5)
    for x in (.45,2.75,5.05):
        arrow(ax,(x+.775,5.22),(x+.775,4.96),color=C["orange"],lw=.9)
    grads=[(r"$G_H$",r"$128\times256$"),(r"$G_A$",r"$128\times256$"),(r"$G_Z$",r"$(P-Y)/m$")]
    gx=[2.3,5.2,8.1]
    for x,(a,b) in zip(gx,grads):
        p=box(ax,x,3.2,1.65,.9,"",fc="#fff3e0",ec=C["orange"]); box_title(ax,x+.825,3.82,a,p)
        text(ax,x+.825,3.45,b,in_box=p,ha="center",va="center",fontsize=10.5)
    arrow(ax,(8.0,3.65),(6.9,3.65),color=C["orange"]); arrow(ax,(5.1,3.65),(4.0,3.65),color=C["orange"])
    for x,label in ((1.2,r"$dW_1=X^{\!\!\top}G_H$"),(7.0,r"$dW_2=A^{\!\!\top}G_Z$")):
        p=box(ax,x,1.15,3.0,.85,"",fc="#f6eadf",ec=C["orange"]); box_title(ax,x+1.5,1.58,label,p,color="#111111")
    arrow(ax,(8.85,3.1),(8.55,2.05),color=C["orange"]); arrow(ax,(3.1,3.1),(3.2,2.05),color=C["orange"])
    dashed_line(ax,(1.2,4.4),(1.2,2.9)); arrow(ax,(1.2,2.9),(2.25,2.9),color=C["orange"],lw=.9)
    arrow(ax,(2.25,2.9),(2.25,2.05),color=C["orange"],lw=.9)
    arrow(ax,(3.55,4.4),(3.55,4.15),color=C["orange"],lw=.9)
    dashed_line(ax,(5.1,4.4),(5.1,2.6)); arrow(ax,(5.1,2.6),(7.5,2.6),color=C["orange"],lw=.9)
    arrow(ax,(7.5,2.6),(7.5,2.05),color=C["orange"],lw=.9)
    text(ax,10.5,4.7,"橙色虚线：保存边",ha="center",fontsize=10.5,color=C["orange"])
    text(ax,6,.75,r"参数梯度从主链分支；$W_1$、$W_2$ 是常驻参数，不计入激活。",ha="center",fontsize=10.5)
    finish(fig,"backprop-mlp-flow")


def figure_flops():
    fig,ax=base_fig("一个 Linear 的三次矩阵乘：前向一次，反向两次",7.6,2.4,(0,12),(0.75,3.5))
    items=[(.5,"前向",r"$X[m,k]\,W[k,n]$","$2mkn$",C["blue"]),(4.3,r"反向 $dW$",r"$X^{\!\!\top}[k,m]\,G[m,n]$","$2mkn$",C["orange"]),(8.1,r"反向 $dX$",r"$G[m,n]\,W^{\!\!\top}[n,k]$","$2mkn$",C["green"])]
    for x,title,formula,cost,color in items:
        p=box(ax,x,1.75,3.25,1.5,"",fc="#f7f7f7",ec=color); box_title(ax,x+1.625,2.82,title,p)
        text(ax,x+1.625,2.3,formula,in_box=p,ha="center",va="center",fontsize=10.5)
        text(ax,x+1.625,2.0,cost,in_box=p,ha="center",va="center",fontsize=10.5,color=color)
    arrow(ax,(3.82,2.5),(4.25,2.5),color="#777777",lw=1.0)
    arrow(ax,(7.62,2.5),(8.05,2.5),color="#777777",lw=1.0)
    text(ax,6,1.05,r"前向 $2mkn$；反向 $2+2=4mkn=2\times$ 前向。",ha="center",fontsize=10.5)
    finish(fig,"backprop-linear-flops")


def figure_eps():
    epses,results=toy_eps()
    fig,ax=plt.subplots(figsize=(7.6,3.4))
    ax.set_title("中心差分相对误差：大步长截断误差，小步长舍入误差",fontweight=600,color="#111111")
    ax.loglog(epses,results[np.float32],"o-",label="fp32",color=C["orange"],lw=1.6)
    ax.loglog(epses,results[np.float64],"o-",label="fp64",color=C["blue"],lw=1.6)
    ax.set_xlabel("epsilon"); ax.set_ylabel("relative error")
    ax.set_xticks([1e-1,1e-3,1e-5,1e-7,1e-9])
    ax.set_xticklabels([r"$10^{-1}$",r"$10^{-3}$",r"$10^{-5}$",r"$10^{-7}$",r"$10^{-9}$"], fontsize=10.5)
    ax.set_yticks([1,1e-4,1e-8,1e-12])
    ax.set_yticklabels(["1",r"$10^{-4}$",r"$10^{-8}$",r"$10^{-12}$"], fontsize=10.5)
    ax.grid(True,which="major",alpha=.2); ax.legend(frameon=False)
    text(ax,3e-6,3e-2,r"fp32 在 $10^{-6}$ 附近已开始舍入失控",fontsize=10.5,color=C["orange"])
    save(fig,"backprop-epsilon-sweep")


def figure_autograd():
    loss,names,edges,leaves=graph_nodes()
    fig,ax=base_fig("Autograd 录下的反向图：从 loss.grad_fn 沿 next_functions 遍历",7.6,5.6,(0,12),(-0.5,11.4))
    positions=[(4.6,10.4),(4.6,8.85),(4.6,7.3),(.9,5.75),(4.6,5.75),(8.3,5.75),
               (4.6,4.2),(8.3,4.2),(2.7,2.65),(6.6,2.65),(6.6,1.25)]
    assert len(names)==len(positions), f"unexpected autograd graph has {len(names)} nodes"
    wdt,hgt=3.25,1.1
    for i,j in edges:
        x1,y1=positions[i]; x2,y2=positions[j]
        arrow(ax,(x1+wdt/2,y1-hgt/2-.03),(x2+wdt/2,y2+hgt/2+.03),color="#777777",lw=.8)
    for (x,y),name,leaf in zip(positions,names,leaves):
        label=name.replace("torch::autograd::","").replace("Backward0","Backward")
        if name.startswith("TBackward"): label="TBackward (W 转置)"
        p=box(ax,x,y-hgt/2,wdt,hgt,"",fc="#f5f8fc",ec=C["blue"],lw=.8)
        if leaf:
            text(ax,x+wdt/2,y+.19,label,in_box=p,ha="center",va="center",fontsize=10.5)
            text(ax,x+wdt/2,y-.19,leaf,in_box=p,ha="center",va="center",fontsize=10.5,color=C["orange"])
        else:
            text(ax,x+wdt/2,y,label,in_box=p,ha="center",va="center",fontsize=10.5)
    text(ax,6,.05,f"loss={loss:.4f}；{len(names)} 个节点；橙色小字 = 该叶子对应的参数",ha="center",fontsize=10.5)
    finish(fig,"backprop-autograd-graph")


def toy_mlp_params():
    """784-256-10 两层 MLP 的参数量（边数 + 偏置数）。"""
    l1 = nn.Linear(784, 256); l2 = nn.Linear(256, 10)
    n = sum(p.numel() for p in list(l1.parameters()) + list(l2.parameters()))
    print(f"W1 {tuple(l1.weight.shape)} + b1 {tuple(l1.bias.shape)}"
          f" + W2 {tuple(l2.weight.shape)} + b2 {tuple(l2.bias.shape)}")
    print(f"784*256 + 256 + 256*10 + 10 = {n}")
    assert n == 203530, n
    print("与图注一致: OK")


def figure_mlp_columns():
    """两层 MLP 三列节点图（784 / 256 / 10，两组全连接边）。"""
    fig, ax = base_fig("两层 MLP：$784\\to256\\to10$，"
                     "两组全连接边就是 $W_1$、$W_2$", h=4.4,
                     xlim=(0, 12), ylim=(0.9, 7.6))
    from matplotlib.patches import Circle
    cols = [(1.55, C["blue"], "#dce9f7", r"输入 $x\in\mathbb{R}^{784}$",
             "$28\\times28$ 像素"),
            (6.0, C["orange"], "#fae6d6", r"$a=\mathrm{ReLU}(xW_1+b_1)$",
             "256 个神经元"),
            (10.45, C["green"], "#e2f0e4", r"$z=aW_2+b_2\to\mathrm{softmax}$",
             "10 个类别")]
    ys = [6.45, 6.0, 5.55, 4.55, 4.1, 3.65]
    for cx, ec_, fc_, main, sub in cols:
        for yv in ys:
            ax.add_patch(Circle((cx, yv), .11, facecolor=fc_,
                                edgecolor=ec_, lw=.9))
        text(ax, cx, 5.05, "$\\vdots$", ha="center", va="center",
             fontsize=10.5, color="#555555")
        text(ax, cx, 7.05, main, ha="center", fontsize=10.5, color="#111111")
        text(ax, cx, 6.78, sub, ha="center", fontsize=10.5, color="#555555")
    for y1 in ys:                                   # 全连接边（可见节点全对全）
        for y2 in ys:
            ax.plot([1.72, 5.83], [y1, y2], color="#5a83a8", lw=.5,
                    alpha=.35, zorder=0)
            ax.plot([6.17, 10.28], [y1, y2], color="#5a83a8", lw=.5,
                    alpha=.35, zorder=0)
    lin_boxes = [(2.3, 2.4), (6.75, 2.4)]
    for bx, by in lin_boxes:
        rect = (bx, by, bx + 3.1, by + 1.0)
        for ln in ax.get_lines():                       # 盒-边相交检查
            xs, ys = ln.get_xdata(), ln.get_ydata()
            assert not _segment_hits_rect(
                (xs[0], ys[0]), (xs[-1], ys[-1]), rect), \
                f"Linear 框压边: {rect}"
        p = box(ax, bx, by, 3.1, 1.0, "", fc="#f7f7f7", ec="#777777", r=.02)
    text(ax, 3.85, 3.02, "第一层 Linear", in_box=ax.patches[-2],
         ha="center", fontsize=10.5, color="#111111")
    text(ax, 3.85, 2.64, r"$W_1:784\times256$，$b_1:256$",
         in_box=ax.patches[-2], ha="center", fontsize=10.5, color="#333333")
    text(ax, 8.3, 3.02, "第二层 Linear", in_box=ax.patches[-1],
         ha="center", fontsize=10.5, color="#111111")
    text(ax, 8.3, 2.64, r"$W_2:256\times10$，$b_2:10$",
         in_box=ax.patches[-1], ha="center", fontsize=10.5, color="#333333")
    text(ax, 6, 1.95, "每个隐藏神经元连着全部 784 个输入（784 条边，"
         "对应 $W_1$ 的一列），每个输出连着全部 256 个隐藏神经元。",
         ha="center", fontsize=10.5, color="#333333")
    text(ax, 6, 1.45, "参数 = 边数 + 偏置数"
         r" $=784\times256+256+256\times10+10=203{,}530$",
         ha="center", fontsize=10.5, color="#111111")
    finish(fig, "backprop-two-layer-mlp")


def figures():
    figure_mlp_columns()
    figure_overview(); figure_scalar(); figure_gate(); figure_fanout(); figure_shapes()
    figure_relu_jacobian(); figure_batched_jacobian(); figure_linear_outer(); figure_softmax()
    figure_mini(); figure_mlp(); figure_flops(); figure_eps(); figure_autograd()
    print("figures written to blog img/in-post")


COMMANDS={"scalar":toy_scalar,"fanout":toy_fanout,"relu-vjp":toy_relu,"jacobian":toy_jacobian,
          "linear":toy_linear,"softmax":toy_softmax,"mini-net":toy_mini,"eps-sweep":toy_eps,
          "autograd-graph":toy_graph,"mlp-params":toy_mlp_params,"figures":figures}
if __name__=="__main__":
    cmd=sys.argv[1] if len(sys.argv)>1 else "scalar"
    if cmd not in COMMANDS: raise SystemExit("commands: "+", ".join(COMMANDS))
    COMMANDS[cmd]()
