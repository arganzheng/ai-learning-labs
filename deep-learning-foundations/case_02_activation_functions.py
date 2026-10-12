"""案例（深度学习基础 02）：六种激活函数的梯度与 MNIST 深层 MLP。

https://arganzheng.life/activation-functions-from-sigmoid-to-swiglu.html

    python case_02_activation_functions.py

固定种子，在同一个 20 × 256 MLP 上比较初始化梯度与 250 步后的准确率；
同时输出激活函数曲线到 out/activation-functions.svg。
"""
import argparse
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from _plot import C, plt, save
from dlf import data

SEED = 2026
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

    fig, axes = plt.subplots(2, 1, figsize=(7.6, 4.6), sharex=True)
    for name, (values, derivatives, color) in curves.items():
        axes[0].plot(x, values, color=color, linewidth=1.5, label=name)
        axes[1].plot(x, derivatives, color=color, linewidth=1.5, label=name)
    axes[0].set_ylabel("激活值")
    axes[0].set_title("六种激活函数及其导数")
    axes[0].legend(ncol=3, loc="upper left")
    axes[1].set_xlabel("输入 x")
    axes[1].set_ylabel("导数")
    for ax in axes:
        ax.axhline(0, color=C["gray"], linewidth=0.6)
        ax.axvline(0, color=C["gray"], linewidth=0.6)
        ax.grid(alpha=0.2)
    save(fig, "activation-functions")


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
    batches = torch.randint(
        TRAIN_LIMIT,
        (STEPS, BATCH_SIZE),
        generator=torch.Generator().manual_seed(SEED + 1),
    )

    print(
        f"=== MNIST：20 × 256 MLP，Kaiming-normal，Adam lr=1e-3，"
        f"{STEPS} 步 × batch {BATCH_SIZE}，固定种子 {SEED} ==="
    )
    print("activation     first-hidden-grad   last-hidden-grad   ratio    test-acc   dead-ReLU")
    for activation_name in ACTIVATIONS:
        torch.manual_seed(SEED)
        model = DeepMLP(activation_name)
        first_grad, last_grad = initial_gradient_norms(
            model, train_x[:BATCH_SIZE], train_y[:BATCH_SIZE]
        )
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
        dead = f"{dead_fraction * 100:6.2f}%" if math.isfinite(dead_fraction) else "    —"
        print(
            f"{activation_name:<12} {first_grad:17.3e} {last_grad:17.3e} "
            f"{first_grad / max(last_grad, 1e-30):8.2e} "
            f"{test_accuracy * 100:8.2f}%   {dead}"
        )

    print("ReLU 死亡单元定义：测试集前 1024 张图上，该隐藏单元对所有输入都输出 0。")
    print("过大学习率示范：")
    dead_relu_demo()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true", help="只生成激活函数曲线图")
    args = parser.parse_args()
    plot_activation_curves()
    if not args.plot_only:
        run_experiment()


if __name__ == "__main__":
    main()
