"""案例（深度学习基础 07）：在完整 text8 上训练 skip-gram + negative sampling。

    .venv/bin/python deep-learning-foundations/case_embedding_word2vec.py

首次运行会从 Matt Mahoney 的 text8 官方页面下载完整语料并缓存。
全程使用 CPU，输出近邻、类比与二维 PCA 图。
"""
from collections import Counter
from pathlib import Path
import random
import time
import urllib.request
import zipfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from _plot import C, plt, save

SEED = 2026
TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
MIN_COUNT = 5
SUBSAMPLE_THRESHOLD = 1e-5
DIMENSION = 100
WINDOW = 5
NEGATIVES = 5
BATCH_SIZE = 16_384
EPOCHS = 3
INITIAL_LR = 0.01
FINAL_LR = 0.002
QUERY_WORDS = (
    "king", "queen", "man", "woman", "computer", "software", "hardware",
    "language",
)
PLOT_WORDS = (
    "king", "queen", "prince", "princess", "man", "woman", "throne",
    "royal", "computer", "software", "hardware", "program", "machine",
    "language", "word", "technology", "internet", "london", "paris",
    "england",
)
DATA_DIR = Path(__file__).with_name("data")


def load_tokens():
    DATA_DIR.mkdir(exist_ok=True)
    archive = DATA_DIR / "text8.zip"
    text_path = DATA_DIR / "text8"
    if not archive.exists():
        print(f"下载 {TEXT8_URL}")
        urllib.request.urlretrieve(TEXT8_URL, archive)
    if not text_path.exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extract("text8", DATA_DIR)
    return text_path.read_bytes().split()


def preprocess(tokens, rng):
    counts = Counter(tokens)
    vocab = [
        word for word, count in counts.most_common()
        if count >= MIN_COUNT
    ]
    word_to_id = {word: index for index, word in enumerate(vocab)}
    frequencies = np.fromiter(
        (counts[word] for word in vocab),
        dtype=np.int64,
        count=len(vocab),
    )
    frequencies = frequencies / len(tokens)
    keep_probability = np.ones(len(vocab), dtype=np.float32)
    frequent = frequencies > SUBSAMPLE_THRESHOLD
    keep_probability[frequent] = (
        np.sqrt(frequencies[frequent] / SUBSAMPLE_THRESHOLD) + 1
    ) * (SUBSAMPLE_THRESHOLD / frequencies[frequent])

    token_ids = np.fromiter(
        (word_to_id.get(word, -1) for word in tokens),
        dtype=np.int32,
        count=len(tokens),
    )
    random_values = rng.random(len(tokens), dtype=np.float32)
    keep = (token_ids >= 0) & (
        random_values < keep_probability[np.maximum(token_ids, 0)]
    )
    ids = token_ids[keep]
    return [word.decode("ascii") for word in vocab], ids, frequencies


def make_training_pairs(ids, rng):
    radii = rng.integers(1, WINDOW + 1, size=len(ids), dtype=np.int8)
    centers, contexts = [], []
    for offset in range(1, WINDOW + 1):
        left = radii[:-offset] >= offset
        centers.append(ids[:-offset][left])
        contexts.append(ids[offset:][left])

        right = radii[offset:] >= offset
        centers.append(ids[offset:][right])
        contexts.append(ids[:-offset][right])

    return np.concatenate(centers), np.concatenate(contexts)


def make_unigram_alias_table(frequencies):
    weights = frequencies.astype(np.float64) ** 0.75
    scaled = weights * len(weights) / weights.sum()
    probabilities = np.empty(len(weights), dtype=np.float32)
    aliases = np.arange(len(weights), dtype=np.int32)
    small = np.flatnonzero(scaled < 1).tolist()
    large = np.flatnonzero(scaled >= 1).tolist()

    while small and large:
        low = small.pop()
        high = large.pop()
        probabilities[low] = scaled[low]
        aliases[low] = high
        scaled[high] -= 1 - scaled[low]
        (small if scaled[high] < 1 else large).append(high)

    for index in small + large:
        probabilities[index] = 1
    return probabilities, aliases


def sample_negatives(probabilities, aliases, count, rng):
    candidates = rng.integers(
        0, len(probabilities), size=count, dtype=np.int32
    )
    coins = rng.random(count, dtype=np.float32)
    return np.where(
        coins < probabilities[candidates], candidates, aliases[candidates]
    ).astype(np.int32, copy=False)


class SkipGram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.input = nn.Embedding(vocab_size, DIMENSION, sparse=True)
        self.output = nn.Embedding(vocab_size, DIMENSION, sparse=True)
        nn.init.uniform_(self.input.weight, -0.5 / DIMENSION, 0.5 / DIMENSION)
        nn.init.zeros_(self.output.weight)

    def forward(self, centers, contexts, negatives):
        center_vectors = self.input(centers)
        positive_vectors = self.output(contexts)
        negative_vectors = self.output(negatives)
        positive_scores = (center_vectors * positive_vectors).sum(dim=1)
        negative_scores = torch.bmm(
            negative_vectors, center_vectors.unsqueeze(2)
        ).squeeze(2)
        return -(
            F.logsigmoid(positive_scores)
            + F.logsigmoid(-negative_scores).sum(dim=1)
        ).mean()


def normalized_embeddings(model):
    weights = model.input.weight.detach().cpu()
    return F.normalize(weights, p=2, dim=1)


def nearest_neighbors(word, embeddings, word_to_id, vocab, exclude=(), limit=5):
    if word not in word_to_id:
        return []
    scores = embeddings @ embeddings[word_to_id[word]]
    for excluded in (*exclude, word):
        if excluded in word_to_id:
            scores[word_to_id[excluded]] = -float("inf")
    indices = torch.topk(scores, limit).indices.tolist()
    return [(vocab[index], float(scores[index])) for index in indices]


def analogy(embeddings, word_to_id, vocab):
    terms = ("king", "man", "woman")
    if not all(word in word_to_id for word in terms):
        return []
    query = (
        embeddings[word_to_id["king"]]
        - embeddings[word_to_id["man"]]
        + embeddings[word_to_id["woman"]]
    )
    query = F.normalize(query, dim=0)
    scores = embeddings @ query
    for word in terms:
        scores[word_to_id[word]] = -float("inf")
    return [(vocab[i], float(scores[i])) for i in torch.topk(scores, 5).indices.tolist()]


def plot_pca(model, word_to_id):
    available = [word for word in PLOT_WORDS if word in word_to_id]
    vectors = model.input.weight.detach().cpu().numpy()[
        [word_to_id[word] for word in available]
    ]
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    points = centered @ components[:2].T
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(points[:, 0], points[:, 1], color=C["blue"], s=35)
    for word, (x, y) in zip(available, points):
        ax.annotate(word, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=9)
    ax.axhline(0, color=C["gray"], linewidth=0.6)
    ax.axvline(0, color=C["gray"], linewidth=0.6)
    ax.grid(alpha=0.2)
    ax.set_title("text8 skip-gram：词向量的 PCA 投影")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    save(fig, "embedding-word2vec")


def show_neighbors(title, embeddings, word_to_id, vocab):
    print(title)
    for word in QUERY_WORDS:
        if word in word_to_id:
            neighbors = nearest_neighbors(word, embeddings, word_to_id, vocab)
            print(
                f"  {word:10} "
                + ", ".join(f"{item} ({score:.3f})" for item, score in neighbors)
            )
    results = analogy(embeddings, word_to_id, vocab)
    if results:
        print(
            "  king - man + woman (原词已排除) = "
            + ", ".join(f"{word} ({score:.3f})" for word, score in results)
        )


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    rng = np.random.default_rng(SEED)
    started = time.perf_counter()

    tokens = load_tokens()
    corpus_size = len(tokens)
    vocab, ids, frequencies = preprocess(tokens, rng)
    del tokens
    word_to_id = {word: index for index, word in enumerate(vocab)}
    centers, contexts = make_training_pairs(ids, rng)
    alias_probabilities, aliases = make_unigram_alias_table(frequencies)
    model = SkipGram(len(vocab))
    initial = normalized_embeddings(model)
    show_neighbors("随机初始化 baseline：", initial, word_to_id, vocab)

    pair_count = len(centers)
    steps_per_epoch = (pair_count + BATCH_SIZE - 1) // BATCH_SIZE
    total_steps = EPOCHS * steps_per_epoch
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=INITIAL_LR)
    order = np.arange(pair_count, dtype=np.int32)
    global_step = 0
    for epoch in range(EPOCHS):
        rng.shuffle(order)
        epoch_loss = 0.0
        for start in range(0, pair_count, BATCH_SIZE):
            sample = order[start:start + BATCH_SIZE]
            batch_size = len(sample)
            progress = global_step / max(total_steps - 1, 1)
            learning_rate = INITIAL_LR + progress * (FINAL_LR - INITIAL_LR)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            batch_centers = torch.from_numpy(centers[sample].astype(np.int64))
            batch_contexts = torch.from_numpy(contexts[sample].astype(np.int64))
            negative_ids = sample_negatives(
                alias_probabilities, aliases, batch_size * NEGATIVES, rng
            ).reshape(batch_size, NEGATIVES)
            negatives = torch.from_numpy(negative_ids.astype(np.int64))

            optimizer.zero_grad(set_to_none=True)
            loss = model(batch_centers, batch_contexts, negatives)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item() * batch_size
            global_step += 1

        print(
            f"epoch {epoch + 1}/{EPOCHS}: loss={epoch_loss / pair_count:.4f}, "
            f"updates={steps_per_epoch:,}"
        )

    trained = normalized_embeddings(model)
    print(
        f"语料：完整 text8 {corpus_size:,} 词；min_count={MIN_COUNT} 后词表 "
        f"{len(vocab):,}；高频词 subsampling 后保留 {len(ids):,} 个词；"
        f"训练对 {pair_count:,}/epoch；维度 {DIMENSION}；"
        f"动态窗口 1–{WINDOW}；负样本 {NEGATIVES}（unigram^0.75 alias table）；"
        f"batch {BATCH_SIZE:,}；epochs {EPOCHS}；总更新 {global_step:,} 步。"
    )
    show_neighbors("训练后：", trained, word_to_id, vocab)
    plot_pca(model, word_to_id)
    print(f"PCA 图：out/embedding-word2vec.svg")
    print(f"CPU 耗时：{time.perf_counter() - started:.1f} 秒")


if __name__ == "__main__":
    main()
