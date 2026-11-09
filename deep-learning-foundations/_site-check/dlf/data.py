"""MNIST (downloaded on first use into ./data/) and a small text corpus for the char LM."""
import glob, gzip, os, urllib.request
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MIRRORS = ["https://ossci-datasets.s3.amazonaws.com/mnist/", "https://storage.googleapis.com/cvdf-datasets/mnist/"]
FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]

def ensure_mnist():
    os.makedirs(DATA, exist_ok=True)
    for f in FILES:
        p = os.path.join(DATA, f)
        if os.path.exists(p): continue
        for m in MIRRORS:
            try:
                print(f"downloading {m}{f} ...")
                urllib.request.urlretrieve(m + f, p + ".part"); os.replace(p + ".part", p); break
            except Exception as e:
                print("  failed:", e)
        else:
            raise RuntimeError(f"could not download {f}; put it into {DATA} by hand")

def load_mnist(kind="train"):
    """returns X [n, 784] float32 in [0, 1] and y [n] int64"""
    ensure_mnist()
    p = "train" if kind == "train" else "t10k"
    with gzip.open(f"{DATA}/{p}-images-idx3-ubyte.gz") as f:
        x = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255.0
    with gzip.open(f"{DATA}/{p}-labels-idx1-ubyte.gz") as f:
        y = np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
    return x, y

def load_mnist_standardized():
    """train/test with per-pixel standardization fitted on the training set (used from post 02 on)"""
    X, y = load_mnist("train"); Xt, yt = load_mnist("test")
    mu, sd = X.mean(0), X.std(0) + 1e-3
    return (X - mu) / sd, y, (Xt - mu) / sd, yt

def load_text(max_chars):
    """Python's own stdlib source as a char-level corpus (always available, no download).
    returns int64 token ids and the vocabulary size."""
    files = sorted(glob.glob(os.path.join(os.path.dirname(os.__file__), "*.py")))
    buf = []
    for f in files:
        try: buf.append(open(f, encoding="utf-8", errors="ignore").read())
        except Exception: pass
        if sum(len(b) for b in buf) > max_chars: break
    s = "".join(buf)[:max_chars]
    chars = sorted(set(s)); stoi = {c: i for i, c in enumerate(chars)}
    return np.array([stoi[c] for c in s], np.int64), len(chars)
