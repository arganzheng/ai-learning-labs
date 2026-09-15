"""Python 使用层（工具箱 01）：流式过一遍 JSONL 语料、dataclass 配置、训练代码里的协议方法、多进程预处理、读 traceback。
https://arganzheng.life/python-in-use-for-algorithm-engineers.html

只用标准库。

    python 00_python_in_use.py            # 全部：jsonl dataclass protocols multiproc traceback
    python 00_python_in_use.py jsonl      # 只跑一个
"""
import hashlib
import json
import random
import re
import sys
import time
import tracemalloc
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from functools import wraps
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from pathlib import Path

OUT = Path(__file__).with_name("out")
CORPUS = OUT / "corpus.jsonl"
N_LINES = 100_000

WORDS = ("模型 训练 数据 梯度 显存 参数 序列 注意力 损失 学习率 batch token layer norm cache "
         "the of and to in a is that for it as with on be at by this from").split()


def make_corpus(path: Path, n: int = N_LINES, seed: int = 0) -> None:
    """合成 n 行 JSONL：{"id", "source", "text"}，约 10% 是重复文本。"""
    rng = random.Random(seed)
    path.parent.mkdir(exist_ok=True)
    pool = [" ".join(rng.choices(WORDS, k=rng.randint(3, 60))) for _ in range(n // 10)]  # 重复来源
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            text = rng.choice(pool) if rng.random() < 0.1 else " ".join(rng.choices(WORDS, k=rng.randint(3, 60)))
            f.write(json.dumps({"id": i, "source": rng.choice(["web", "book", "code"]), "text": text}, ensure_ascii=False) + "\n")


# ---------------- 1. 流式过一遍语料：生成器 vs 一次读进内存 ----------------
def read_all(path: Path) -> list[dict]:
    """把整个文件读成一个 list——内存与文件大小成正比。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def iter_jsonl(path: Path):
    """生成器：一次只在内存里放一行。"""
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def clean(records, min_words: int = 5):
    """过滤 + 去重也写成生成器，可以串起来。"""
    seen: set[str] = set()
    for r in records:
        if len(r["text"].split()) < min_words:
            continue
        h = hashlib.md5(r["text"].encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        yield r


def exp_jsonl():
    print("=== 1. 流式过一遍语料：生成器 vs 一次读进内存 ===")
    if not CORPUS.exists():
        make_corpus(CORPUS)
    size_mb = CORPUS.stat().st_size / 2**20
    print(f"语料 {CORPUS.name}: {N_LINES:,} 行, {size_mb:.1f} MB")

    tracemalloc.start()
    records = read_all(CORPUS)
    n_all = len(records)
    _, peak_list = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del records

    tracemalloc.start()
    by_source, lengths = Counter(), Counter()
    kept = 0
    for r in clean(iter_jsonl(CORPUS)):
        kept += 1
        by_source[r["source"]] += 1
        lengths[min(len(r["text"].split()) // 10 * 10, 50)] += 1
    _, peak_gen = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"一次读进 list: {n_all:,} 条, 峰值内存 {peak_list / 2**20:.1f} MB  （≈ 文件大小 × {peak_list / CORPUS.stat().st_size:.1f}）")
    print(f"生成器流式:   峰值内存 {peak_gen / 2**20:.1f} MB  （只有去重的哈希集合在涨）")
    print(f"过滤 + 去重后保留 {kept:,} / {n_all:,} 条; 按来源: {dict(sorted(by_source.items()))}")
    print("按长度分桶(词数下界): " + ", ".join(f"{k}+: {v}" for k, v in sorted(lengths.items())))
    print()


# ---------------- 2. dataclass 配置 ----------------
@dataclass
class TrainConfig:
    model: str = "Qwen/Qwen2.5-0.5B"
    lr: float = 2e-5
    batch_size: int = 8
    max_steps: int = 1000
    warmup: int = 50
    lora_targets: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])  # 可变默认值要用 factory

    def __post_init__(self):
        assert self.warmup <= self.max_steps, f"warmup {self.warmup} > max_steps {self.max_steps}"


def exp_dataclass():
    print("=== 2. dataclass 配置：默认值、覆盖、序列化 ===")
    cfg = TrainConfig()
    print("默认:", cfg)
    overrides = {"lr": 1e-4, "max_steps": 200}                    # 命令行 / JSON 里来的覆盖
    cfg2 = replace(cfg, **overrides)                              # 生成新对象，原来的不动
    print("覆盖后:", cfg2.lr, cfg2.max_steps, "| 原来的:", cfg.lr, cfg.max_steps)
    print("存进 run 目录的 config.json:", json.dumps(asdict(cfg2), ensure_ascii=False))
    print("两个默认对象的 lora_targets 是否同一个 list:", TrainConfig().lora_targets is TrainConfig().lora_targets)
    try:
        TrainConfig(warmup=500, max_steps=200)
    except AssertionError as e:
        print("非法配置在构造时就报错:", e)
    print()


# ---------------- 3. 训练代码里的协议方法 ----------------
class ToyDataset:
    """PyTorch 的 Dataset 只要求两个方法：__len__ 和 __getitem__。"""

    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return {"input_ids": [ord(c) % 128 for c in self.texts[i]], "label": len(self.texts[i]) % 2}


def collate(items: list[dict]) -> dict:
    """把一批样本拼成 batch：右侧 pad 到最长。"""
    T = max(len(x["input_ids"]) for x in items)
    return {"input_ids": [x["input_ids"] + [0] * (T - len(x["input_ids"])) for x in items],
            "attention_mask": [[1] * len(x["input_ids"]) + [0] * (T - len(x["input_ids"])) for x in items],
            "labels": [x["label"] for x in items]}


def loader(ds, batch_size: int, shuffle: bool, seed: int = 0):
    """DataLoader 的骨架：一个生成器，按索引取样本、攒够一批就 collate 后 yield。"""
    idx = list(range(len(ds)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for s in range(0, len(idx), batch_size):
        yield collate([ds[i] for i in idx[s:s + batch_size]])


class ToyModel:
    """nn.Module 的 __call__：model(x) 等于 model.forward(x) 加前后钩子。"""

    def __init__(self):
        self.calls = 0

    def __call__(self, batch):
        self.calls += 1                  # 真实的 __call__ 在这里跑 forward hooks
        return self.forward(batch)

    def forward(self, batch):
        return [sum(row) / max(1, sum(m)) for row, m in zip(batch["input_ids"], batch["attention_mask"])]


def timed(fn):
    """装饰器：函数包函数。@timed 等价于 fn = timed(fn)。"""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        print(f"  [{fn.__name__} 用时 {time.perf_counter() - t0:.3f}s]")
        return out
    return wrapper


@contextmanager
def seeded(seed: int):
    """上下文管理器：进入时定 seed，退出时恢复随机状态——torch.no_grad() / autocast 是同样的形状。"""
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


@timed
def one_epoch(model, ds, batch_size=4):
    n = 0
    for batch in loader(ds, batch_size, shuffle=True):
        model(batch)
        n += len(batch["labels"])
    return n


def exp_protocols():
    print("=== 3. 训练代码里的协议方法：__len__/__getitem__、生成器、__call__、装饰器、with ===")
    ds = ToyDataset(["attention is all you need", "loss", "the memory ledger", "bf16", "rope", "kv cache", "sft", "dpo", "grpo"])
    print(f"len(ds) = {len(ds)}; ds[0] = {ds[0]}")
    first = next(loader(ds, batch_size=4, shuffle=False))
    print(f"第一个 batch: input_ids 形状 [{len(first['input_ids'])}, {len(first['input_ids'][0])}], labels = {first['labels']}")
    model = ToyModel()
    n = one_epoch(model, ds)
    print(f"一个 epoch 看了 {n} 个样本, model 被调用 {model.calls} 次 (= ceil(9 / 4))")
    with seeded(42):
        a = [random.random() for _ in range(3)]
    with seeded(42):
        b = [random.random() for _ in range(3)]
    print(f"seeded(42) 两次得到相同的数: {a == b}; 退出后随机状态已恢复")
    print()


# ---------------- 4. 多进程预处理 ----------------
TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenize_count(line: str) -> int:
    """CPU 密集的一小步：正则切词 + 哈希。"""
    toks = TOKEN_RE.findall(json.loads(line)["text"])
    hashlib.sha256(" ".join(toks).encode()).hexdigest()
    return len(toks)


def exp_multiproc():
    print("=== 4. 多进程预处理：串行 vs 线程池 vs 进程池 ===")
    if not CORPUS.exists():
        make_corpus(CORPUS)
    lines = CORPUS.read_text(encoding="utf-8").splitlines()
    import os
    nproc = min(8, os.cpu_count() or 1)

    t0 = time.perf_counter()
    total = sum(map(tokenize_count, lines))
    t_serial = time.perf_counter() - t0

    with ThreadPool(nproc) as pool:
        t0 = time.perf_counter()
        total_t = sum(pool.map(tokenize_count, lines, chunksize=2000))
        t_thread = time.perf_counter() - t0

    with Pool(nproc) as pool:
        t0 = time.perf_counter()
        total_p = sum(pool.map(tokenize_count, lines, chunksize=2000))
        t_proc = time.perf_counter() - t0

    assert total == total_t == total_p
    print(f"{len(lines):,} 行, 共 {total:,} 个 token; {nproc} 个 worker")
    print(f"串行:   {t_serial:.2f}s")
    print(f"线程池: {t_thread:.2f}s  （{t_serial / t_thread:.1f}×，GIL 让 CPU 密集的线程几乎不并行）")
    print(f"进程池: {t_proc:.2f}s  （{t_serial / t_proc:.1f}×，进程各有一个解释器，代价是启动与序列化）")
    print()


# ---------------- 5. 读 traceback ----------------
def project(x: list[list[float]], w: list[list[float]]) -> list[list[float]]:
    """x: [B, d_in], w: [d_in, d_out]。"""
    d_in = len(w)
    for row in x:
        assert len(row) == d_in, f"shape mismatch: x row has {len(row)} features, w expects {d_in}"
    return [[sum(a * b for a, b in zip(row, col)) for col in zip(*w)] for row in x]


def forward(batch):
    h = project(batch, [[0.1] * 4 for _ in range(3)])   # w: [3, 4]
    return h


def train_step(batch):
    return forward(batch)


def exp_traceback():
    print("=== 5. 读 traceback：从下往上 ===")
    import traceback
    try:
        train_step([[1.0, 2.0, 3.0], [4.0, 5.0]])         # 第二行只有 2 个特征
    except AssertionError:
        tb = traceback.format_exc().rstrip().splitlines()
        print("\n".join(tb))
        print(f"\n共 {len([l for l in tb if l.strip().startswith('File')])} 层调用; 最后一行是错误本身, 往上第一帧是出错的位置, 再往上是它怎么被调到的")
    print()


EXPS = {"jsonl": exp_jsonl, "dataclass": exp_dataclass, "protocols": exp_protocols, "multiproc": exp_multiproc, "traceback": exp_traceback}

if __name__ == "__main__":
    for name in sys.argv[1:] or EXPS:
        EXPS[name]()
