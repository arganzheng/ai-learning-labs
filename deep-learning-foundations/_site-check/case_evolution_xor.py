"""Perceptron, ADALINE, and a two-layer MLP on XOR.

    python3 case_evolution_xor.py
"""
import statistics

import numpy as np

from _plot import C, plt, save


X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR = np.array([0.0, 1.0, 1.0, 0.0])
AND = np.array([0.0, 0.0, 0.0, 1.0])
EPOCHS = 100
ADALINE_EPOCHS = 500
MLP_STEPS = 5000
SEEDS = range(20)


def step(values):
    return (values >= 0).astype(float)


def train_perceptron(targets):
    weights = np.zeros(2)
    bias = 0.0
    errors = []
    states = {}
    cycle = None
    for epoch in range(EPOCHS):
        mistakes = 0
        for x, target in zip(X, targets):
            prediction = step(np.array([weights @ x + bias]))[0]
            if prediction != target:
                delta = target - prediction
                weights += delta * x
                bias += delta
                mistakes += 1
        errors.append(mistakes)
        state = (*weights.tolist(), bias)
        if state in states and cycle is None:
            cycle = (states[state] + 1, epoch - states[state])
        else:
            states[state] = epoch
    return weights, bias, errors, cycle


def train_adaline():
    weights = np.zeros(2)
    bias = 0.0
    losses = []
    for _ in range(ADALINE_EPOCHS):
        predictions = X @ weights + bias
        residual = predictions - XOR
        losses.append(float(np.mean(residual**2)))
        weights -= 0.1 * (2.0 / len(X)) * (X.T @ residual)
        bias -= 0.1 * (2.0 / len(X)) * residual.sum()
    residual = X @ weights + bias - XOR
    loss = float(np.mean(residual**2))
    losses.append(loss)
    coefficients, *_ = np.linalg.lstsq(np.column_stack((X, np.ones(len(X)))), XOR, rcond=None)
    closed_predictions = np.column_stack((X, np.ones(len(X)))) @ coefficients
    closed_loss = float(np.mean((closed_predictions - XOR) ** 2))
    return weights, bias, loss, losses, coefficients, closed_predictions, closed_loss


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def mlp_forward(x, parameters):
    w1, b1, w2, b2 = parameters
    hidden = np.tanh(x @ w1 + b1)
    logits = hidden @ w2 + b2
    return hidden, logits, sigmoid(logits)


def mlp_loss(logits, targets):
    return float(np.mean(np.logaddexp(0.0, logits) - targets * logits))


def train_mlp(hidden_size, seed):
    rng = np.random.default_rng(seed)
    bound1 = 1.0 / np.sqrt(X.shape[1])
    bound2 = 1.0 / np.sqrt(hidden_size)
    parameters = [
        rng.uniform(-bound1, bound1, (X.shape[1], hidden_size)),
        rng.uniform(-bound1, bound1, hidden_size),
        rng.uniform(-bound2, bound2, (hidden_size, 1)),
        rng.uniform(-bound2, bound2, 1),
    ]
    targets = XOR[:, None]
    losses = []
    for _ in range(MLP_STEPS):
        hidden, logits, probabilities = mlp_forward(X, parameters)
        losses.append(mlp_loss(logits, targets))
        dlogits = (probabilities - targets) / len(X)
        w1, b1, w2, b2 = parameters
        d_w2 = hidden.T @ dlogits
        d_b2 = dlogits.sum(axis=0)
        d_hidden = dlogits @ w2.T
        d_pre_activation = d_hidden * (1.0 - hidden**2)
        d_w1 = X.T @ d_pre_activation
        d_b1 = d_pre_activation.sum(axis=0)
        parameters = [
            w1 - 0.5 * d_w1,
            b1 - 0.5 * d_b1,
            w2 - 0.5 * d_w2,
            b2 - 0.5 * d_b2,
        ]
    hidden, logits, probabilities = mlp_forward(X, parameters)
    losses.append(mlp_loss(logits, targets))
    accuracy = float(np.mean((probabilities[:, 0] >= 0.5) == XOR))
    return parameters, losses, accuracy, losses[-1]


def plot_case(perceptron, adaline, selected_mlp):
    p_weights, p_bias = perceptron
    a_weights, a_bias = adaline
    mlp_parameters, mlp_seed, mlp_losses = selected_mlp
    grid = np.linspace(-0.2, 1.2, 181)
    xx, yy = np.meshgrid(grid, grid)
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))

    fig = plt.figure(figsize=(14, 7.5))
    layout = fig.add_gridspec(2, 6, height_ratios=(1.05, 1.0))
    axes = [
        fig.add_subplot(layout[0, 0:2]),
        fig.add_subplot(layout[0, 2:4]),
        fig.add_subplot(layout[0, 4:6]),
    ]
    surfaces = [
        step(grid_points @ p_weights + p_bias).reshape(xx.shape),
        (grid_points @ a_weights + a_bias).reshape(xx.shape),
        mlp_forward(grid_points, mlp_parameters)[2][:, 0].reshape(xx.shape),
    ]
    titles = [
        "感知机：XOR 最后一轮",
        "ADALINE：线性输出",
        f"两层 MLP：H=4，seed={mlp_seed}",
    ]
    for index, (ax, surface, title) in enumerate(zip(axes, surfaces, titles)):
        if index == 0:
            ax.contourf(xx, yy, surface, levels=[-0.5, 0.5, 1.5],
                        colors=["#e8eef7", "#f8e9e6"], alpha=0.95)
            score = (grid_points @ p_weights + p_bias).reshape(xx.shape)
        else:
            if index == 1:
                surface = np.round(surface, 3)
            ax.contourf(xx, yy, surface, levels=np.linspace(0, 1, 11),
                        cmap="RdYlBu_r", vmin=0, vmax=1, alpha=0.8)
            score = surface
        if index != 1 and np.nanmin(score) < 0.5 < np.nanmax(score):
            ax.contour(xx, yy, score, levels=[0.5], colors=["#222222"], linewidths=1.2)
        for label, color, marker in ((0, C["blue"], "o"), (1, C["red"], "X")):
            chosen = XOR == label
            ax.scatter(X[chosen, 0], X[chosen, 1], c=color, marker=marker, s=58,
                       edgecolors="white", linewidths=0.8, zorder=3)
        ax.set_title(title)
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.grid(alpha=0.16)
    axes[1].text(0.03, 0.04, "4 个预测均约 0.5", transform=axes[1].transAxes,
                 fontsize=8, color="#222222")

    errors_ax = fig.add_subplot(layout[1, 0:3])
    adaline_ax = fig.add_subplot(layout[1, 3:6])
    p_and = train_perceptron(AND)[2]
    p_xor = train_perceptron(XOR)[2]
    epochs = np.arange(1, EPOCHS + 1)
    errors_ax.plot(epochs, p_and, color=C["green"], label="AND")
    errors_ax.plot(epochs, p_xor, color=C["red"], label="XOR")
    errors_ax.set_title("感知机：每轮更新次数")
    errors_ax.set_xlabel("epoch")
    errors_ax.set_ylabel("误分数")
    errors_ax.legend(frameon=False, ncol=2)
    errors_ax.grid(alpha=0.2)

    _, _, _, adaline_losses, _, _, _ = train_adaline()
    steps = np.arange(len(adaline_losses))
    sampled = np.arange(0, len(adaline_losses), 5)
    if sampled[-1] != len(adaline_losses) - 1:
        sampled = np.append(sampled, len(adaline_losses) - 1)
    mlp_sampled = np.linspace(0, len(mlp_losses) - 1, 501).astype(int)
    adaline_ax.plot(steps[sampled], np.asarray(adaline_losses)[sampled],
                    color=C["orange"], label="ADALINE MSE")
    adaline_ax.plot(mlp_sampled, np.asarray(mlp_losses)[mlp_sampled],
                    color=C["blue"], label="MLP H=4 BCE")
    adaline_ax.set_title("平方误差与交叉熵")
    adaline_ax.set_xlabel("更新步")
    adaline_ax.set_ylabel("loss")
    adaline_ax.legend(frameon=False)
    adaline_ax.grid(alpha=0.2)
    save(fig, "dl-evolution-xor")


def main():
    and_weights, and_bias, and_errors, _ = train_perceptron(AND)
    xor_weights, xor_bias, xor_errors, xor_cycle = train_perceptron(XOR)
    and_epoch = next((i + 1 for i, errors in enumerate(and_errors) if errors == 0), None)
    print("=== XOR：固定顺序 (0,0), (0,1), (1,0), (1,1) ===")
    print(f"感知机 AND：η=1，w=b=0，100 epochs；误分首次为 0 的 epoch={and_epoch}")
    print(
        "感知机 XOR：η=1，w=b=0，100 epochs；"
        f"每轮误分最小值={min(xor_errors)}，有无 0 误分={'有' if 0 in xor_errors else '无'}"
    )
    print(
        f"  最终 w=[{xor_weights[0]:.0f}, {xor_weights[1]:.0f}], b={xor_bias:.0f}；"
        f"权重状态循环={xor_cycle if xor_cycle else '100 轮内未发现'}"
    )
    print(
        f"  AND 最终 w=[{and_weights[0]:.0f}, {and_weights[1]:.0f}], "
        f"b={and_bias:.0f}；误分={and_errors[-1]}"
    )

    adaline_weights, adaline_bias, adaline_loss, adaline_losses, coefficients, predictions, closed_loss = train_adaline()
    print(f"ADALINE XOR：batch GD η=0.1，500 epochs，MSE={adaline_loss:.10f}")
    print(f"  最终 w=[{adaline_weights[0]:.10f}, {adaline_weights[1]:.10f}], b={adaline_bias:.10f}")
    print(
        "  lstsq [w1, w2, b]="
        f"[{coefficients[0]:.10f}, {coefficients[1]:.10f}, {coefficients[2]:.10f}]，"
        f"MSE={closed_loss:.10f}，预测={np.array2string(predictions, precision=6)}"
    )

    print(
        f"MLP XOR：tanh → sigmoid，BCE，η=0.5，{MLP_STEPS} full-batch steps；"
        "uniform ±1/√fan_in，seed=0…19"
    )
    selected_mlp = None
    for hidden_size in (2, 4):
        results = [train_mlp(hidden_size, seed) for seed in SEEDS]
        accuracies = [item[2] for item in results]
        losses = [item[3] for item in results]
        successes = sum(accuracy == 1.0 for accuracy in accuracies)
        if hidden_size == 4:
            seed = next(seed for seed, item in zip(SEEDS, results) if item[2] == 1.0)
            parameters, losses_curve, accuracy, final_loss = results[seed]
            selected_mlp = (parameters, seed, losses_curve)
        print(
            f"  H={hidden_size}: 100% accuracy={successes}/20 seeds，"
            f"final BCE median={statistics.median(losses):.8f}，max={max(losses):.8f}"
        )
    if selected_mlp is None:
        raise RuntimeError("No H=4 seed reached 100% accuracy; no decision-boundary example available.")
    print(f"  图中 H=4 成功 seed={selected_mlp[1]}，final BCE={final_loss:.8f}")
    plot_case((xor_weights, xor_bias), (adaline_weights, adaline_bias), selected_mlp)


if __name__ == "__main__":
    main()
