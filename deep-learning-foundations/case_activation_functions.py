"""案例（深度学习基础 03）：激活函数族的曲线与 MNIST 深层 MLP。

https://arganzheng.life/activation-functions-from-sigmoid-to-swiglu.html

    python case_activation_functions.py

三个固定种子，在同一个 20 × 256 MLP 上比较初始化梯度与 250 步后的
准确率；同时输出函数族曲线到 out/activation-*.svg。
"""
import argparse
import math
import statistics

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib.patches import FancyBboxPatch

from _plot import C, plt, save
from dlf import data

SEEDS = (2026, 2027, 2028)
HIDDEN_LAYERS = 20
WIDTH = 256
TRAIN_LIMIT = 12000
STEPS = 250
BATCH_SIZE = 128
ACTIVATIONS = {
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "ReLU": nn.ReLU,
    "LeakyReLU": lambda: nn.LeakyReLU(negative_slope=0.01),
    "GELU": nn.GELU,
    "SiLU": nn.SiLU,
}


class DeepMLP(nn.Module):
    def __init__(self, activation_name):
        super().__init__()
        self.hidden = nn.ModuleList()
        in_features = 784
        for _ in range(HIDDEN_LAYERS):
            layer = nn.Linear(in_features, WIDTH)
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
            self.hidden.append(layer)
            in_features = WIDTH
        self.output = nn.Linear(WIDTH, 10)
        nn.init.kaiming_normal_(self.output.weight, nonlinearity="relu")
        nn.init.zeros_(self.output.bias)
        self.activation = ACTIVATIONS[activation_name]()

    def hidden_activations(self, x):
        activations = []
        for layer in self.hidden:
            x = self.activation(layer(x))
            activations.append(x)
        return activations

    def forward(self, x):
        for layer in self.hidden:
            x = self.activation(layer(x))
        return self.output(x)


def plot_activation_curves():
    x = np.linspace(-6, 6, 1201)
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    cdf = np.array([0.5 * (1 + math.erf(value / math.sqrt(2))) for value in x])
    pdf = np.exp(-(x**2) / 2) / math.sqrt(2 * math.pi)
    curves = {
        "sigmoid": (sigmoid, sigmoid * (1 - sigmoid), C["blue"]),
        "tanh": (tanh, 1 - tanh**2, C["orange"]),
        "ReLU": (np.maximum(x, 0), np.where(x > 0, 1, 0), C["green"]),
        "LeakyReLU": (np.where(x > 0, x, 0.01 * x), np.where(x > 0, 1, 0.01), C["purple"]),
        "GELU": (x * cdf, cdf + x * pdf, C["red"]),
        "SiLU": (x * sigmoid, sigmoid + x * sigmoid * (1 - sigmoid), C["teal"]),
    }

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True, sharey=True)
    for ax, (name, (values, derivatives, color)) in zip(axes.flat, curves.items()):
        ax.plot(x, values, color=color, linewidth=1.6, label="函数")
        ax.plot(x, derivatives, color=color, linewidth=1.4, linestyle="--", label="导数")
        ax.set_title(name)
        ax.legend(loc="upper left", frameon=False, fontsize=8)
        ax.axhline(0, color=C["gray"], linewidth=0.6)
        ax.axvline(0, color=C["gray"], linewidth=0.6)
        ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("激活值 / 导数")
    axes[1, 0].set_ylabel("激活值 / 导数")
    for ax in axes[1]:
        ax.set_xlabel("输入 x")
    save(fig, "activation-functions")


def _format_axes(ax, title, ylabel="Activation"):
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Input x", fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(labelsize=14)
    ax.axhline(0, color=C["gray"], linewidth=0.7)
    ax.axvline(0, color=C["gray"], linewidth=0.7)
    ax.grid(alpha=0.2)


def _plot_function_and_derivative(name, x, functions, derivatives, colors, legends):
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 5.2), sharex=True, constrained_layout=False,
    )
    for label, values, derivative, color in zip(legends, functions, derivatives, colors):
        axes[0].plot(x, values, color=color, linewidth=2.4, label=label)
        axes[1].plot(x, derivative, color=color, linewidth=2.4, linestyle="--", label=label)
    _format_axes(axes[0], f"{name}: functions")
    _format_axes(axes[1], f"{name}: derivatives", "Derivative")
    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, legends, frameon=False, fontsize=15,
        ncol=min(2, len(legends)), loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    return fig, axes


def plot_sigmoid_tanh():
    x = np.linspace(-6, 6, 1201)
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    fig, axes = _plot_function_and_derivative(
        "Sigmoid and tanh", x,
        [sigmoid, tanh], [sigmoid * (1 - sigmoid), 1 - tanh**2],
        [C["blue"], C["orange"]], ["sigmoid", "tanh"],
    )
    for ax in axes:
        ax.axvspan(-6, -3, color=C["gray"], alpha=0.12)
        ax.axvspan(3, 6, color=C["gray"], alpha=0.12)
    axes[1].scatter([0, 0], [0.25, 1], color=[C["blue"], C["orange"]], zorder=4)
    axes[1].annotate("sigmoid: 0.25", (0, 0.25), xytext=(0.45, 0.08), fontsize=15)
    axes[1].annotate("tanh: 1", (0, 1), xytext=(0.45, 0.84), fontsize=15)
    save(fig, "activation-sigmoid-tanh")


def plot_relu():
    x = np.linspace(-5, 5, 1001)
    values = np.maximum(x, 0)
    derivative = np.where(x > 0, 1, 0)
    fig, axes = _plot_function_and_derivative(
        "ReLU", x, [values], [derivative], [C["green"]], ["ReLU"],
    )
    for ax in axes:
        ax.axvspan(-5, 0, color=C["red"], alpha=0.10)
    axes[0].text(-4.7, 3.6, "zero-gradient region (x < 0)", fontsize=15)
    save(fig, "activation-relu")


def plot_relu_variants():
    x = np.linspace(-5, 5, 1001)
    leaky = np.where(x > 0, x, 0.1 * x)
    prelu = np.where(x > 0, x, 0.25 * x)
    elu = np.where(x > 0, x, np.expm1(x))
    relu = np.maximum(x, 0)
    functions = [leaky, prelu, elu, relu]
    derivatives = [
        np.where(x > 0, 1, 0.1),
        np.where(x > 0, 1, 0.25),
        np.where(x > 0, 1, np.exp(x)),
        np.where(x > 0, 1, 0),
    ]
    colors = [C["purple"], C["blue"], C["orange"], C["gray"]]
    labels = ["Leaky (α=0.1)", "PReLU (α=0.25 example)",
              "ELU (α=1)", "ReLU reference"]
    fig, axes = _plot_function_and_derivative(
        "ReLU variants", x, functions, derivatives, colors, labels,
    )
    for ax in axes:
        ax.lines[3].set_linewidth(1.2)
    save(fig, "activation-relu-variants")


def plot_smooth_activations():
    # 读者反馈：gelu_new 与精确 GELU 在函数图上完全重合（最大差 4.73e-4），
    # 改成三栏：函数、导数、tanh 近似的误差。只用颜色区分曲线。
    vermilion = "#D55E00"
    green = "#009E73"
    blue = "#0072B2"
    purple = "#CC79A7"

    x = np.linspace(-5, 5, 6001)
    cdf = np.array([0.5 * (1 + math.erf(value / math.sqrt(2))) for value in x])
    pdf = np.exp(-(x**2) / 2) / math.sqrt(2 * math.pi)
    sigmoid = 1 / (1 + np.exp(-x))
    gelu = x * cdf
    silu = x * sigmoid
    relu = np.maximum(x, 0)
    gelu_new_inner = math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)
    gelu_new = 0.5 * x * (1 + np.tanh(gelu_new_inner))
    error = gelu_new - gelu

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0))
    colors = [vermilion, green, blue]
    labels = ["GELU", "SiLU", "ReLU reference"]
    functions = [gelu, silu, relu]
    derivatives = [
        cdf + x * pdf,
        sigmoid + x * sigmoid * (1 - sigmoid),
        np.where(x > 0, 1, 0),
    ]
    for values, label, color in zip(functions, labels, colors):
        axes[0].plot(x, values, color=color, linewidth=2.2, label=label)
    for values, label, color in zip(derivatives, labels, colors):
        axes[1].plot(x, values, color=color, linewidth=2.2, label=label)
    axes[2].plot(x, error * 1e4, color=purple, linewidth=2.2)

    _format_axes(axes[0], "Smooth activations: functions")
    _format_axes(axes[1], "Smooth activations: derivatives", "Derivative")
    _format_axes(axes[2], "tanh approximation error", "gelu_new - GELU (x1e-4)")
    axes[0].legend(frameon=True, framealpha=0.9, fontsize=13, loc="upper left")

    gelu_min = int(np.argmin(gelu))
    silu_min = int(np.argmin(silu))
    for index, label, color in (
        (gelu_min, "GELU: (-0.752, -0.170)", vermilion),
        (silu_min, "SiLU: (-1.278, -0.278)", green),
    ):
        is_gelu = label.startswith("GELU")
        y = gelu[index] if is_gelu else silu[index]
        axes[0].scatter([x[index]], [y], color=color, zorder=4)
        text_y = 0.55 if is_gelu else 0.12
        axes[0].annotate(
            label, (x[index], y), xytext=(-4.5, text_y), fontsize=13,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 1},
        )
    print(f"  gelu_new - GELU 在 [-5,5] 最大绝对误差 "
          f"{np.abs(error).max():.3e}（x1e-4 轴）")
    save(fig, "activation-smooth")


def plot_glu_family():
    # Okabe-Ito 色盲友好配色；用户要求只用颜色区分，统一实线
    blue = "#0072B2"
    grey = "#7F7F7F"
    orange = "#E69F00"
    reddish_purple = "#CC79A7"
    green = "#009E73"

    fig, (diagram, gates) = plt.subplots(
        1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1.15, 1]},
    )
    diagram.set_xlim(0, 10)
    diagram.set_ylim(0, 10)
    diagram.axis("off")
    diagram.set_title("FFN projections", fontsize=11)

    def box(ax, center, text, width=2.4, height=1.05):
        x, y = center
        patch = FancyBboxPatch(
            (x - width / 2, y - height / 2), width, height,
            boxstyle="round,pad=0.12", facecolor="#f4f7fb",
            edgecolor="#7a8ca5", linewidth=1.1,
        )
        ax.add_patch(patch)
        ax.text(x, y, text, ha="center", va="center", fontsize=14)

    def arrow(ax, start, end):
        ax.annotate("", xy=end, xytext=start,
                    arrowprops={"arrowstyle": "->", "color": C["gray"], "lw": 1})

    def code_name(ax, x, y, text):
        ax.text(x, y, text, ha="center", va="top", fontsize=10.5,
                color=C["gray"], style="italic")

    diagram.text(0.2, 8.9, "Ordinary FFN", fontsize=16, weight="bold")
    diagram.text(0.55, 7.5, "x", ha="center", va="center", fontsize=14)
    box(diagram, (2.3, 7.5), "W1\nd → $d_{ff}$")
    box(diagram, (5.1, 7.5), "activation")
    box(diagram, (8.0, 7.5), "W2\n$d_{ff}$ → d")
    arrow(diagram, (0.85, 7.5), (1.15, 7.5))
    arrow(diagram, (3.45, 7.5), (3.95, 7.5))
    arrow(diagram, (6.25, 7.5), (6.85, 7.5))

    diagram.text(0.2, 6.1, "Gated FFN", fontsize=16, weight="bold")
    diagram.text(0.55, 4.2, "x", ha="center", va="center", fontsize=14)
    box(diagram, (2.3, 5.0), "W gate\nd → $d_g$")
    box(diagram, (2.3, 3.1), "V value\nd → $d_g$")
    box(diagram, (5.2, 4.05), "g(xW)\n⊙ xV", width=2.6, height=1.15)
    box(diagram, (8.1, 4.05), "W2\n$d_g$ → d")
    code_name(diagram, 2.3, 4.28, "w1 / gate_proj")
    code_name(diagram, 2.3, 2.38, "w3 / up_proj")
    code_name(diagram, 8.1, 3.3, "w2 / down_proj")
    arrow(diagram, (0.85, 4.4), (1.1, 4.85))
    arrow(diagram, (0.85, 4.0), (1.1, 3.35))
    arrow(diagram, (3.5, 5.0), (3.9, 4.45))
    arrow(diagram, (3.5, 3.1), (3.9, 3.65))
    arrow(diagram, (6.5, 4.05), (6.95, 4.05))
    diagram.text(5.1, 1.35,
                 "$d_g \\approx \\frac{2}{3}\\, d_{ff}$ keeps parameters "
                 "matched ($3\\,d\\,d_g \\approx 2\\,d\\,d_{ff}$)",
                 ha="center", fontsize=12.5, color=C["gray"])

    x = np.linspace(-5, 5, 1001)
    sigmoid = 1 / (1 + np.exp(-x))
    curves = [
        (sigmoid, "GLU: sigmoid", blue),
        (x, "Bilinear: x", grey),
        (np.maximum(x, 0), "ReGLU: ReLU", orange),
        (x * np.array([0.5 * (1 + math.erf(v / math.sqrt(2))) for v in x]),
         "GEGLU: GELU", reddish_purple),
        (x * sigmoid, "SwiGLU: SiLU", green),
    ]
    gates.set_title("Gate functions", fontsize=16)
    for values, label, color in curves:
        gates.plot(x, values, color=color, linewidth=2.2, label=label)
    gates.axhline(0, color=C["gray"], linewidth=0.7)
    gates.axvline(0, color=C["gray"], linewidth=0.7)
    gates.set_xlabel("Input x", fontsize=15)
    gates.set_ylabel("Gate output", fontsize=15)
    gates.tick_params(labelsize=15)
    gates.grid(alpha=0.2)
    gates.legend(frameon=True, framealpha=0.9, fontsize=13, loc="upper left")
    save(fig, "activation-glu-family")


def plot_function_families():
    plot_sigmoid_tanh()
    plot_relu()
    plot_relu_variants()
    plot_smooth_activations()
    plot_glu_family()


@torch.no_grad()
def accuracy(model, x, y):
    correct = 0
    for start in range(0, len(x), 512):
        logits = model(x[start : start + 512])
        correct += (logits.argmax(dim=1) == y[start : start + 512]).sum().item()
    return correct / len(x)


@torch.no_grad()
def relu_dead_fraction(model, x):
    dead = sum(
        (activation == 0).all(dim=0).sum().item()
        for activation in model.hidden_activations(x)
    )
    return dead / (HIDDEN_LAYERS * WIDTH)


def initial_gradient_norms(model, x, y):
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(x), y).backward()
    return model.hidden[0].weight.grad.norm().item(), model.hidden[-1].weight.grad.norm().item()


def dead_relu_demo():
    x = torch.tensor([[1.0], [2.0]])
    target = torch.ones_like(x)
    layer = nn.Linear(1, 1)
    with torch.no_grad():
        layer.weight.fill_(2.0)
        layer.bias.zero_()
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.6)
    before = F.relu(layer(x)).detach().flatten().tolist()
    loss = 0.5 * (F.relu(layer(x)) - target).square().sum()
    loss.backward()
    optimizer.step()
    after = F.relu(layer(x)).detach().flatten().tolist()
    stuck_loss = 0.5 * (F.relu(layer(x)) - target).square().sum()
    optimizer.zero_grad(set_to_none=True)
    stuck_loss.backward()
    next_gradient = layer.weight.grad.abs().sum().item() + layer.bias.grad.abs().sum().item()
    print(
        "  单个 ReLU 单元：lr=0.6，w 2.00→-2.20，b 0.00→-2.40，"
        f"输出 {before}→{after}，loss 5.00→1.00，下一步梯度 {next_gradient:.0f}"
    )


def run_experiment():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    train_x, train_y, test_x, test_y = data.load_mnist_standardized()
    train_x = torch.from_numpy(train_x[:TRAIN_LIMIT])
    train_y = torch.from_numpy(train_y[:TRAIN_LIMIT])
    test_x = torch.from_numpy(test_x)
    test_y = torch.from_numpy(test_y)
    print(
        f"=== MNIST：20 × 256 MLP，Kaiming-normal，Adam lr=1e-3，"
        f"{STEPS} 步 × batch {BATCH_SIZE}，种子 {', '.join(map(str, SEEDS))} ==="
    )
    print(
        "activation     first-hidden-grad   last-hidden-grad   ratio"
        "    test-acc mean [min–max]    dead-ReLU mean [min–max]"
    )
    gradients = {}
    results = {name: [] for name in ACTIVATIONS}
    for seed in SEEDS:
        batches = torch.randint(
            TRAIN_LIMIT,
            (STEPS, BATCH_SIZE),
            generator=torch.Generator().manual_seed(seed + 1),
        )
        for activation_name in ACTIVATIONS:
            torch.manual_seed(seed)
            model = DeepMLP(activation_name)
            first_grad, last_grad = initial_gradient_norms(
                model, train_x[:BATCH_SIZE], train_y[:BATCH_SIZE]
            )
            if seed == SEEDS[0]:
                gradients[activation_name] = (first_grad, last_grad)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            for indices in batches:
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(train_x[indices]), train_y[indices])
                loss.backward()
                optimizer.step()
            test_accuracy = accuracy(model, test_x, test_y)
            dead_fraction = (
                relu_dead_fraction(model, test_x[:1024])
                if activation_name == "ReLU"
                else float("nan")
            )
            results[activation_name].append((test_accuracy, dead_fraction))

    for activation_name, runs in results.items():
        first_grad, last_grad = gradients[activation_name]
        accuracies = [run[0] for run in runs]
        acc_text = (
            f"{statistics.mean(accuracies) * 100:5.2f}% "
            f"[{min(accuracies) * 100:.2f}, {max(accuracies) * 100:.2f}]"
        )
        dead_values = [run[1] for run in runs if math.isfinite(run[1])]
        dead_text = (
            f"{statistics.mean(dead_values) * 100:5.2f}% "
            f"[{min(dead_values) * 100:.2f}, {max(dead_values) * 100:.2f}]"
            if dead_values
            else "—"
        )
        print(
            f"{activation_name:<12} {first_grad:17.3e} {last_grad:17.3e} "
            f"{first_grad / max(last_grad, 1e-30):8.2e} "
            f"{acc_text:25} {dead_text}"
        )

    print(f"初始化梯度列取种子 {SEEDS[0]}。")
    print("ReLU 死亡单元定义：测试集前 1024 张图上，该隐藏单元对所有输入都输出 0。")
    print("过大学习率示范：")
    dead_relu_demo()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true", help="只生成激活函数曲线图")
    args = parser.parse_args()
    plot_activation_curves()
    plot_function_families()
    if not args.plot_only:
        run_experiment()


if __name__ == "__main__":
    main()
