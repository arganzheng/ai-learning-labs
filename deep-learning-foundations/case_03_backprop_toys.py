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


def dashed_line(ax, start, end, color=C["orange"], lw=.9):
    assert_line_clear(ax, start, end)
    ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=lw, ls="--")


def arrow(ax, start, end, color="#555555", lw=1.2, style="-|>"):
    assert_line_clear(ax, start, end)
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style,
                                 mutation_scale=12, color=color, lw=lw,
                                 shrinkA=1, shrinkB=1))


def base_fig(title, w=7.6, h=4.3, xlim=(0, 12), ylim=(0, 7)):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    ax.set_title(title, fontweight=600, color="#111111", pad=10)
    return fig, ax


def node(ax, x, y, label, value="", grad="", fc="#f5f8fc", ec=C["blue"], w=1.45, h=1.1):
    p = box(ax, x, y, w, h, "", fc=fc, ec=ec, lw=1.0)
    box_title(ax, x+w/2, y+h*.73, label, p, color="#111111")
    if value:
        text(ax, x+w/2, y+h*.48, value, in_box=p, ha="center", va="center", fontsize=8.5, color=C["green"])
    if grad:
        text(ax, x+w/2, y+h*.2, grad, in_box=p, ha="center", va="center", fontsize=8.5, color=C["orange"])
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
    x=np.array([.6,-1.2]); W1=np.array([[.5,-.3],[.8,.4]]); b1=np.array([1.0,1.0])
    W2=np.array([[.7,-.5],[.2,.6]]); b2=np.array([.05,-.1]); target=0
    h=x@W1+b1; a=np.maximum(h,0); z=a@W2+b2
    p=np.exp(z-z.max()); p/=p.sum(); loss=-np.log(p[target])
    gz=p.copy(); gz[target]-=1
    dw2=np.outer(a,gz); db2=gz; ga=W2@gz; gh=ga*(h>0); dw1=np.outer(x,gh); db1=gh
    return locals()


def toy_mini():
    d=mini_values()
    for key in ("x","W1","b1","h","a","W2","b2","z","p"):
        print(f"{key}={np.array2string(d[key],precision=6)}")
    print(f"target={d['target']}; loss={d['loss']:.9f}")
    for key in ("gz","dw2","db2","ga","gh","dw1","db1"):
        print(f"{key}={np.array2string(d[key],precision=6)}")
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
    print(f"中心差分 eps=1e-6：dW1、dW2 最大相对误差={max(errs):.3e}")


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
    return loss.item(),[node.name() for node in nodes],edges


def toy_graph():
    loss,order,edges=graph_nodes()
    print(f"loss={loss:.9f}; BFS 遍历 loss.grad_fn.next_functions：")
    for i,name in enumerate(order): print(f"{i:02d} {name}")
    print("edges="+", ".join(f"{a}->{b}" for a,b in edges))
    print(f"unique autograd nodes={len(order)}")


def figure_overview():
    fig,ax=base_fig("反向传播做什么：前向算 loss，反向一次得到所有参数梯度",7.6,3.3,(0,12),(0,5.3))
    labels=[("x","输入"),("H","Linear"),("A","ReLU"),("Z","Linear"),("L","loss")]
    xs=[.65,3.0,5.35,7.7,10.05]; y=3.0
    for (name,sub),x in zip(labels,xs):
        p=box(ax,x,y,1.35,1.0,"",fc="#dce9f7",ec=C["blue"])
        box_title(ax,x+.675,y+.68,name,p,color="#111111")
        text(ax,x+.675,y+.25,sub,in_box=p,ha="center",va="center",fontsize=8.5)
    for i in range(4): arrow(ax,(xs[i]+1.4,y+.5),(xs[i+1]-.08,y+.5),color=C["blue"])
    text(ax,6,2.25,"前向：逐层计算",ha="center",va="center",fontsize=10,color=C["blue"])
    gx=[("dW1","对 W1 的导数"),("dW2","对 W2 的导数")]
    for x,(name,sub) in zip((3.0,7.7),gx):
        p=box(ax,x,.45,1.8,.9,"",fc="#fff3e0",ec=C["orange"])
        box_title(ax,x+.9,.98,name,p,color="#111111")
        text(ax,x+.9,.7,sub,in_box=p,ha="center",va="center",fontsize=7.5)
    arrow(ax,(10.72,2.95),(8.65,1.45),color=C["orange"])
    arrow(ax,(10.65,2.9),(4.4,1.45),color=C["orange"])
    text(ax,6,.12,"反向：沿计算图把 loss 对参数的导数送回每一层",ha="center",va="center",fontsize=9.5)
    save(fig,"backprop-overview")


def figure_scalar():
    fig,ax=base_fig("一个标量：每条边传回上游梯度乘局部导数",7.6,3.6,(0,12),(0,6))
    names=[("x","2"),("z=w*x+b","5"),("e=z-y","1"),("L=e^2","1")]
    xs=[.5,3.3,6.2,9.1]; y=3.4
    for i,((n,v),x) in enumerate(zip(names,xs)):
        node(ax,x,y,n,v,("dL/dx=6" if i==0 else "dL/dz=2" if i==1 else "dL/de=2" if i==2 else ""),w=2.0,h=1.55)
    local=["w=3","1","2e=2"]
    for i,l in enumerate(local):
        arrow(ax,(xs[i]+2.05,y+.75),(xs[i+1]-.05,y+.75),color=C["blue"])
        text(ax,(xs[i]+xs[i+1]+2)/2,y+1.9,l,ha="center",va="center",fontsize=8.5,color=C["green"])
    text(ax,6,1.35,"x=2, w=3, b=-1, y=4；绿色=前向值，橙色=反向梯度。",ha="center",va="center",fontsize=9)
    text(ax,6,.85,"参数与输入的梯度同样存在：dL/dw=4, dL/db=2, dL/dx=6；w+0.001 时 L=1.004004，有限差分验证 dL/dw≈4。",ha="center",va="center",fontsize=9)
    save(fig,"backprop-scalar-graph")


def figure_gate():
    fig,ax=base_fig("反向节点只做一件事：上游梯度乘局部导数",7.6,3.0,(0,12),(0,5))
    p=box(ax,4.2,2.05,3.6,1.25,"",fc="#f5f8fc",ec=C["blue"])
    box_title(ax,6,2.92,"一个计算节点",p,color="#111111")
    text(ax,6,2.52,"局部导数 f'(x)=3",in_box=p,ha="center",va="center",fontsize=9)
    arrow(ax,(1.5,2.65),(4.1,2.65),color=C["orange"]); text(ax,2.8,2.0,"上游梯度 2",ha="center",fontsize=9)
    arrow(ax,(7.9,2.65),(10.5,2.65),color=C["orange"]); text(ax,9.2,2.0,"下游梯度 6",ha="center",fontsize=9)
    text(ax,6,.9,"例：2+3，上游 4，输出梯度 (4,4)；2×3，上游 4，输出 (12,8)；max(2,3)，输出 (0,4)。",ha="center",va="center",fontsize=8.5)
    text(ax,6,.48,"加法原样分发；乘法乘另一个输入；max / ReLU 只送到被选中的一路。",ha="center",va="center",fontsize=8.5)
    save(fig,"backprop-local-vjp")


def figure_fanout():
    fig,ax=base_fig("一个量走了两条路：两路梯度在 x 处相加",7.6,3.2,(0,12),(0,5))
    p=box(ax,.6,2.0,1.5,1.0,"",fc="#dce9f7",ec=C["blue"]); box_title(ax,1.35,2.66,"x=2",p)
    p1=box(ax,4.2,3.25,2.1,.85,"",fc="#f5f8fc",ec="#777777"); box_title(ax,5.25,3.8,"x*y",p1)
    p2=box(ax,4.2,1.0,2.1,.85,"",fc="#f5f8fc",ec="#777777"); box_title(ax,5.25,1.55,"+x",p2)
    p3=box(ax,9.3,2.0,1.7,1.0,"",fc="#fff3e0",ec=C["orange"]); box_title(ax,10.15,2.66,"f=8",p3)
    arrow(ax,(2.15,2.7),(4.1,3.55)); arrow(ax,(2.15,2.3),(4.1,1.45)); arrow(ax,(6.35,3.55),(9.2,2.75)); arrow(ax,(6.35,1.45),(9.2,2.25))
    text(ax,7.75,3.65,"沿 x*y：贡献 3",ha="center",fontsize=8.5,color=C["orange"])
    text(ax,7.75,1.25,"沿 +x：贡献 1",ha="center",fontsize=8.5,color=C["orange"])
    text(ax,6,.35,"df/dx = 3 + 1 = 4",ha="center",fontsize=10,color=C["orange"])
    save(fig,"backprop-fanout")


def figure_shapes():
    fig,ax=base_fig("梯度与被求导的量同形；向量对向量才得到 Jacobian",7.6,3.6,(0,12),(0,6))
    cards=[(.35,"标量 L 对向量 x","x=(1,2,3)\nL=sum(x_i^2)","dL/dx=(2,4,6)\n形状 3；与 x 同形",C["blue"]),
           (4.25,"标量 L 对矩阵 W","W=[[1,2],[3,4]]\nL=sum(W_ij^2)","dL/dW=[[2,4],[6,8]]\n2x2；与 W 同形",C["orange"]),
           (8.15,"向量 y 对向量 x","x 为 3 维\ny 为 2 维","J 为 2x3\n第 i 行是 y_i 的梯度",C["green"])]
    for x,title,ex,out,c in cards:
        p=box(ax,x,2.0,3.45,2.55,"",fc="#f7f7f7",ec=c)
        box_title(ax,x+1.725,4.15,title,p,color="#111111")
        text(ax,x+1.725,3.55,ex,in_box=p,ha="center",va="center",fontsize=8.7)
        text(ax,x+1.725,2.62,out,in_box=p,ha="center",va="center",fontsize=8.4,color=c)
    text(ax,6,.95,"每格 g_ij 表示：W_ij 增加 epsilon，L 约增加 g_ij*epsilon。",ha="center",fontsize=8.5)
    text(ax,6,.55,"W -= lr*dW 要逐元素对应，所以参数梯度必须和 W 一样大。",ha="center",fontsize=8.5)
    save(fig,"backprop-gradient-shapes")


def figure_relu_jacobian():
    fig,ax=base_fig("ReLU 的 Jacobian 是对角掩码：VJP 等于逐元素乘",7.6,3.1,(0,12),(0,5))
    text(ax,2.0,3.15,"J = diag(0,1,1)",ha="center",fontsize=11,color=C["blue"])
    mat=[[0,0,0],[0,1,0],[0,0,1]]
    for r in range(3):
        for c in range(3):
            fc="#eeeeee" if mat[r][c]==0 else "#dce9f7"
            ax.add_patch(Rectangle((.8+c*.72,2.1-r*.65),.62,.55,facecolor=fc,edgecolor="#999999"))
            text(ax,1.11+c*.72,2.37-r*.65,str(mat[r][c]),ha="center",va="center",fontsize=9)
    text(ax,5.2,2.5,"g_y=(0.3, -0.2, 0.5)",ha="center",fontsize=10)
    arrow(ax,(3.35,2.45),(4.2,2.45))
    arrow(ax,(6.3,2.45),(7.1,2.45))
    text(ax,9.1,2.5,"g_x=J^T g_y\n(0, -0.2, 0.5)",ha="center",va="center",fontsize=10,color=C["orange"])
    text(ax,6,.55,"同一计算可写成：g_x = g_y 逐元素乘 (0,1,1)，不必存完整矩阵。",ha="center",fontsize=9)
    save(fig,"backprop-relu-jacobian")


def figure_batched_jacobian():
    fig,ax=base_fig("batch Linear 的 Jacobian：分块对角，VJP 直接成为矩阵乘",7.6,3.5,(0,12),(0,6))
    text(ax,2.8,4.8,"m=3 个样本；每个输入、输出 2 维",ha="center",fontsize=9)
    # 6x6 block grid, two-wide blocks
    x0,y0,cell=.6,1.15,.38
    for r in range(6):
        for c in range(6):
            block=r//2==c//2
            fc="#dce9f7" if block else "#eeeeee"
            ax.add_patch(Rectangle((x0+c*cell,y0+(5-r)*cell),cell-.02,cell-.02,facecolor=fc,edgecolor="white",lw=.4))
            if block: text(ax,x0+c*cell+(cell-.02)/2,y0+(5-r)*cell+(cell-.02)/2,"Wt",ha="center",va="center",fontsize=5.5)
    text(ax,2.0,.55,"6x6 Jacobian：对角块 W^T，其余为 0",ha="center",fontsize=8.5)
    arrow(ax,(3.5,2.3),(4.6,2.3))
    p=box(ax,4.8,1.7,2.45,1.5,"",fc="#f5f8fc",ec=C["blue"]); box_title(ax,6.025,2.72,"G W^T",p)
    text(ax,6.025,2.15,"直接做 batched VJP",in_box=p,ha="center",va="center",fontsize=8.5)
    text(ax,9.4,3.5,"本文第一层 batch 128",ha="center",fontsize=9)
    vals=[("Jacobian",2.2,"13 GB",C["red"]),("dX",.65,"392 KiB",C["blue"]),("W1",.72,"784 KiB",C["green"])]
    for i,(lab,width,value,c) in enumerate(vals):
        yy=2.9-i*.6
        text(ax,8.4,yy,lab,ha="right",va="center",fontsize=8)
        ax.add_patch(Rectangle((8.6,yy-.12),width,.22,facecolor=c,edgecolor="none"))
        text(ax,11.7,yy,value,ha="right",va="center",fontsize=8)
    text(ax,9.7,.55,"长度按对数缩放：13 GB、392 KiB、784 KiB",ha="center",fontsize=8.5)
    save(fig,"backprop-no-full-jacobian")


def figure_linear_outer():
    fig,ax=base_fig("Linear 的权重梯度：两个样本外积相加",7.6,3.2,(0,12),(0,5))
    blocks=[(.55,2.65,"样本 1：x1^T g1","(1,2,-1)^T (.2,-.3)"),
            (.55,1.0,"样本 2：x2^T g2","(.5,-1,2)^T (1,.5)"),
            (8.2,1.85,"逐元素相加","dW = X^T G")]
    for x,y,title,sub in blocks:
        p=box(ax,x,y,3.0,1.2,"",fc="#f7f7f7",ec=C["blue"] if y>2 else C["green"],lw=1.0)
        box_title(ax,x+1.5,y+.82,title,p,color="#111111")
        text(ax,x+1.5,y+.36,sub,in_box=p,ha="center",va="center",fontsize=7.8)
    arrow(ax,(3.6,3.25),(8.05,2.65),color=C["blue"])
    arrow(ax,(3.6,1.55),(8.05,2.15),color=C["green"])
    text(ax,6,.45,"每个样本都用过 W，因此每个样本贡献一份外积；batch 维度沿样本求和。",ha="center",fontsize=8.5)
    save(fig,"backprop-linear-outer-products")


def figure_softmax():
    z=np.array([2.,1.,.1]); p=np.exp(z-z.max()); p/=p.sum(); grad=p-np.array([1.,0.,0.])
    fig,ax=base_fig("softmax + 交叉熵：正确类上推，其余类按概率下压",7.6,3.2,(0,12),(0,5))
    xs=[3,6,9]
    for i,x in enumerate(xs):
        h=.5+abs(grad[i])*2.6; y=1.55
        c=C["green"] if i==0 else C["orange"]
        ax.add_patch(Rectangle((x-.52,y),1.04,h,facecolor=c,alpha=.28,edgecolor=c,lw=1.2))
        text(ax,x,y+h+.28,f"类 {i}",ha="center",fontsize=9)
        text(ax,x,y+.18,f"p={p[i]:.3f}\ng={grad[i]:+.3f}",ha="center",va="bottom",fontsize=8)
    text(ax,6,.6,"dL/dz = p - onehot(y)；梯度下降会提高正确类 logit，降低其他类 logit。",ha="center",fontsize=9)
    save(fig,"backprop-softmax-ce")


def figure_mini():
    d=mini_values()
    fig,ax=base_fig("2-2-2 迷你网络：前向值在上，反向梯度在下",7.6,3.4,(0,12),(0,6))
    labels=[("x",np.array2string(d['x'],precision=1)),("h",np.array2string(d['h'],precision=2)),("a",np.array2string(d['a'],precision=2)),("z",np.array2string(d['z'],precision=2)),("L",f"{d['loss']:.3f}")]
    xs=[.3,2.7,5.1,7.5,9.9]
    for i,((name,val),x) in enumerate(zip(labels,xs)):
        p=box(ax,x,2.7,1.8,1.4,"",fc="#dce9f7",ec=C["blue"]); box_title(ax,x+.9,3.72,name,p)
        text(ax,x+.9,3.1,val,in_box=p,ha="center",va="center",fontsize=8)
        if i<4: arrow(ax,(x+1.84,3.4),(xs[i+1]-.05,3.4),color=C["blue"])
    grads=[("dW1",np.array2string(d['dw1'],precision=2)),("dW2",np.array2string(d['dw2'],precision=2))]
    for x,(name,val) in zip((2.7,7.5),grads):
        p=box(ax,x,.5,2.4,1.6,"",fc="#fff3e0",ec=C["orange"]); box_title(ax,x+1.2,1.68,name,p)
        text(ax,x+1.2,1.02,val,in_box=p,ha="center",va="center",fontsize=6)
    arrow(ax,(10.75,2.65),(8.75,2.16),color=C["orange"]); arrow(ax,(10.65,2.6),(3.95,2.16),color=C["orange"])
    text(ax,6,.2,"有限差分逐元素验证 dW1、dW2；最大相对误差由 toy 输出。",ha="center",fontsize=8.5)
    save(fig,"backprop-mini-network")


def figure_mlp():
    fig,ax=base_fig("784-256-10：前向、主梯度链、参数梯度分三排",7.6,4.2,(0,12),(0,7))
    xs=[.45,2.75,5.05,7.35,9.65]
    fw=[("X","[128,784]"),("H","[128,256]"),("A","[128,256]"),("Z","[128,10]"),("L","scalar")]
    for x,(a,b) in zip(xs,fw):
        p=box(ax,x,5.22,1.55,1.05,"",fc="#dce9f7",ec=C["blue"]); box_title(ax,x+.775,5.96,a,p)
        text(ax,x+.775,5.5,b,in_box=p,ha="center",va="center",fontsize=7.2)
    for i in range(4): arrow(ax,(xs[i]+1.6,5.72),(xs[i+1]-.05,5.72),color=C["blue"])
    tags=[(.45,"存 X"),(2.75,"存 H 符号"),(5.05,"存 A")]
    for x,label in tags:
        p=box(ax,x,4.42,1.55,.52,"",fc="#fae6d6",ec=C["orange"],lw=.8)
        text(ax,x+.775,4.68,label,in_box=p,ha="center",va="center",fontsize=8)
    for x in (.45,2.75,5.05):
        arrow(ax,(x+.775,5.22),(x+.775,4.96),color=C["orange"],lw=.9,style="-|>")
    grads=[("G_H","[128,256]"),("G_A","[128,256]"),("G_Z","[128,10]")]
    gx=[2.3,5.2,8.1]
    for x,(a,b) in zip(gx,grads):
        p=box(ax,x,3.2,1.65,.9,"",fc="#fff3e0",ec=C["orange"]); box_title(ax,x+.825,3.82,a,p)
        text(ax,x+.825,3.45,b,in_box=p,ha="center",va="center",fontsize=7.5)
    arrow(ax,(8.0,3.65),(6.9,3.65),color=C["orange"]); arrow(ax,(5.1,3.65),(4.0,3.65),color=C["orange"])
    for x,label in ((1.2,"dW1 = X^T G_H"),(7.0,"dW2 = A^T G_Z")):
        p=box(ax,x,1.15,3.0,.85,"",fc="#f6eadf",ec=C["orange"]); box_title(ax,x+1.5,1.58,label,p,color="#111111")
    arrow(ax,(8.85,3.1),(8.55,2.05),color=C["orange"]); arrow(ax,(3.1,3.1),(3.2,2.05),color=C["orange"])
    # Save-to-consumer elbows route through the open columns, never through gradient boxes.
    arrow(ax,(1.22,4.40),(.28,4.32),color=C["orange"],lw=.9,style="-|>")
    dashed_line(ax,(.28,4.32),(.28,1.58))
    arrow(ax,(.28,1.58),(1.16,1.58),color=C["orange"],lw=.9,style="-|>")
    arrow(ax,(4.32,4.38),(3.55,4.15),color=C["orange"],lw=.9,style="-|>")
    arrow(ax,(6.6,4.38),(7.5,4.32),color=C["orange"],lw=.9,style="-|>")
    dashed_line(ax,(7.5,4.32),(7.5,2.14))
    arrow(ax,(7.5,2.14),(7.5,2.06),color=C["orange"],lw=.9,style="-|>")
    text(ax,10.5,4.7,"橙色虚线：保存边",ha="center",fontsize=7.5,color=C["orange"])
    text(ax,6,.45,"参数梯度从主链分支；W1、W2 是常驻参数，不计入激活。",ha="center",fontsize=8.5)
    save(fig,"backprop-mlp-flow")


def figure_flops():
    fig,ax=base_fig("一个 Linear 的三次矩阵乘：前向一次，反向两次",7.6,3.1,(0,12),(0,5))
    items=[(.5,"前向","X[m,k] × W[k,n]","2mkn",C["blue"]),(4.3,"反向 dW","X^T[k,m] × G[m,n]","2mkn",C["orange"]),(8.1,"反向 dX","G[m,n] × W^T[n,k]","2mkn",C["green"])]
    for x,title,formula,cost,color in items:
        p=box(ax,x,1.65,3.25,1.55,"",fc="#f7f7f7",ec=color); box_title(ax,x+1.625,2.75,title,p)
        text(ax,x+1.625,2.25,formula,in_box=p,ha="center",va="center",fontsize=8)
        text(ax,x+1.625,1.88,cost,in_box=p,ha="center",va="center",fontsize=8.5,color=color)
    arrow(ax,(3.82,2.4),(4.25,2.4),color="#777777",lw=1.0)
    arrow(ax,(7.62,2.4),(8.05,2.4),color="#777777",lw=1.0)
    text(ax,6,.6,"forward 2mkn；backward 4mkn = 2 × forward。",ha="center",fontsize=9)
    save(fig,"backprop-linear-flops")


def figure_eps():
    epses,results=toy_eps()
    fig,ax=plt.subplots(figsize=(7.6,3.8))
    ax.set_title("中心差分相对误差：大步长截断误差，小步长舍入误差",fontweight=600,color="#111111")
    ax.loglog(epses,results[np.float32],"o-",label="fp32",color=C["orange"],lw=1.6)
    ax.loglog(epses,results[np.float64],"o-",label="fp64",color=C["blue"],lw=1.6)
    ax.set_xlabel("epsilon"); ax.set_ylabel("relative error")
    ax.set_xticks([1e-1,1e-3,1e-5,1e-7,1e-9])
    ax.set_xticklabels(["1e-1","1e-3","1e-5","1e-7","1e-9"])
    ax.set_yticks([1,1e-4,1e-8,1e-12])
    ax.set_yticklabels(["1","1e-4","1e-8","1e-12"])
    ax.grid(True,which="major",alpha=.2); ax.legend(frameon=False)
    text(ax,3e-6,3e-2,"fp32 在 1e-6 附近已开始舍入失控",fontsize=8,color=C["orange"])
    save(fig,"backprop-epsilon-sweep")


def figure_autograd():
    loss,names,edges=graph_nodes()
    fig,ax=base_fig("Autograd 录下的反向图：从 loss.grad_fn 沿 next_functions 遍历",7.6,4.3,(0,12),(0,7.5))
    positions=[(5.0,6.45),(5.0,5.45),(5.0,4.45),(1.0,3.45),(5.0,3.45),(9.0,3.45),
               (5.0,2.45),(9.0,2.45),(3.0,1.45),(7.0,1.45),(7.0,.45)]
    assert len(names)==len(positions), f"unexpected autograd graph has {len(names)} nodes"
    width,height=2.55,.62
    for i,j in edges:
        x1,y1=positions[i]; x2,y2=positions[j]
        arrow(ax,(x1+width/2,y1-height/2-.03),(x2+width/2,y2+height/2+.03),color="#777777",lw=.8)
    for (x,y),name in zip(positions,names):
        label=name.replace("torch::autograd::","").replace("Backward0","Backward")
        p=box(ax,x,y-height/2,width,height,"",fc="#f5f8fc",ec=C["blue"],lw=.8)
        text(ax,x+width/2,y,label,in_box=p,ha="center",va="center",fontsize=7)
    text(ax,2.0,.15,f"loss={loss:.4f}；{len(names)} 个节点",ha="left",va="center",fontsize=8.5)
    save(fig,"backprop-autograd-graph")


def figures():
    figure_overview(); figure_scalar(); figure_gate(); figure_fanout(); figure_shapes()
    figure_relu_jacobian(); figure_batched_jacobian(); figure_linear_outer(); figure_softmax()
    figure_mini(); figure_mlp(); figure_flops(); figure_eps(); figure_autograd()
    print("figures written to blog img/in-post")


COMMANDS={"scalar":toy_scalar,"fanout":toy_fanout,"relu-vjp":toy_relu,"jacobian":toy_jacobian,
          "linear":toy_linear,"softmax":toy_softmax,"mini-net":toy_mini,"eps-sweep":toy_eps,
          "autograd-graph":toy_graph,"figures":figures}
if __name__=="__main__":
    cmd=sys.argv[1] if len(sys.argv)>1 else "scalar"
    if cmd not in COMMANDS: raise SystemExit("commands: "+", ".join(COMMANDS))
    COMMANDS[cmd]()
