"""案例（深度学习基础 07）：在 text8 上训练 skip-gram + negative sampling。

    .venv/bin/python deep-learning-foundations/case_embedding_word2vec.py

首次运行会从 Matt Mahoney 的 text8 官方页面下载语料；截取固定前缀并缓存。
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
CORPUS_TOKENS = 750_000
MIN_COUNT = 5
MAX_VOCAB = 20_000
DIMENSION = 64
WINDOW = 2
NEGATIVES = 5
BATCH_SIZE = 1024
STEPS = 1500
QUERY_WORDS = ("king", "queen", "man", "woman", "computer", "language")
PLOT_WORDS = (
    "king", "queen", "man", "woman", "prince", "princess", "royal",
    "computer", "software", "language", "word", "city", "country",
    "london", "paris",
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
    with text_path.open(encoding="utf-8") as f:
        return f.read(CORPUS_TOKENS * 16).split()[:CORPUS_TOKENS]


def make_training_pairs(tokens):
    counts = Counter(tokens)
    vocab = [
        word for word, count in counts.most_common(MAX_VOCAB)
        if count >= MIN_COUNT
    ]
    word_to_id = {word: index for index, word in enumerate(vocab)}
    ids = np.fromiter(
        (word_to_id[word] for word in tokens if word in word_to_id),
        dtype=np.int64,
    )
    centers, contexts = [], []
    for offset in range(1, WINDOW + 1):
        centers.extend((ids[:-offset], ids[offset:]))
        contexts.extend((ids[offset:], ids[:-offset]))
    center_ids = torch.from_numpy(np.concatenate(centers))
    context_ids = torch.from_numpy(np.concatenate(contexts))
    frequencies = np.bincount(ids, minlength=len(vocab))
    noise = torch.from_numpy(frequencies.astype(np.float64) ** 0.75).float()
    noise /= noise.sum()
    return vocab, word_to_id, ids, center_ids, context_ids, noise


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
            print(f"  {word:10} " + ", ".join(f"{item} ({score:.3f})" for item, score in neighbors))
    results = analogy(embeddings, word_to_id, vocab)
    if results:
        print(
            "  king - man + woman = "
            + ", ".join(f"{word} ({score:.3f})" for word, score in results)
        )


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    started = time.perf_counter()

    tokens = load_tokens()
    vocab, word_to_id, ids, centers, contexts, noise = make_training_pairs(tokens)
    model = SkipGram(len(vocab))
    initial = normalized_embeddings(model)
    show_neighbors("随机初始化 baseline：", initial, word_to_id, vocab)

    generator = torch.Generator().manual_seed(SEED + 1)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.025)
    pair_count = len(centers)
    for step in range(STEPS):
        sample = torch.randint(pair_count, (BATCH_SIZE,), generator=generator)
        batch_centers = centers[sample]
        batch_contexts = contexts[sample]
        negatives = torch.multinomial(
            noise, BATCH_SIZE * NEGATIVES, replacement=True, generator=generator
        ).view(BATCH_SIZE, NEGATIVES)
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch_centers, batch_contexts, negatives)
        loss.backward()
        optimizer.step()

    trained = normalized_embeddings(model)
    print(
        f"语料：text8 前 {len(tokens):,} 个词；保留词表 {len(vocab):,}；"
        f"正样本对 {pair_count:,}；维度 {DIMENSION}；窗口 ±{WINDOW}；"
        f"负样本 {NEGATIVES}；更新 {STEPS:,} 步。"
    )
    show_neighbors("训练后：", trained, word_to_id, vocab)
    plot_pca(model, word_to_id)
    print(f"PCA 图：out/embedding-word2vec.svg")
    print(f"CPU 耗时：{time.perf_counter() - started:.1f} 秒")


if __name__ == "__main__":
    main()
