"""pretrain_e2e 各步共用：路径、语料读写、简单的语言识别、计时。"""
import gzip
import json
import os
import re
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
os.makedirs(DATA, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

WET_BASE = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-10/segments/1707947473347.0/wet/CC-MAIN-20240220211055-20240221001055-{:05d}.warc.wet.gz"
N_WET = 2                                                     # 用几个 WET 文件（每个约 150 MB 压缩、约 3 万个网页）
WET_FILES = [os.path.join(DATA, f"CC-MAIN-2024-10-{i:05d}.warc.wet.gz") for i in range(N_WET)]


def ensure_wets(partial_ok=False):
    """Common Crawl 的 WET 文件（已从 HTML 抽出正文）。没有就断点续传下载（download.py），完成后去掉 .part 后缀。"""
    from download import download
    paths = []
    for i, f in enumerate(WET_FILES):
        if os.path.exists(f):
            paths.append(f); continue
        part = f + ".part"
        if partial_ok and os.path.exists(part):
            paths.append(part); continue
        url = WET_BASE.format(i)
        print(f"下载 {url}\n  → {f}（约 107 MB，慢的网络要几十分钟；中断后重跑会续传）")
        paths.append(download(url, f))
    return paths


def read_wet(path, limit=None):
    """迭代 WET 里的每篇文档：{'url', 'lang_hint', 'text'}。WET 是 Common Crawl 已经从 HTML 抽出正文的格式。"""
    n = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        header, body, in_body = {}, [], False
        try:
            for line in f:
                if line.startswith("WARC/1.0"):
                    if header.get("WARC-Type") == "conversion" and body:
                        yield {"url": header.get("WARC-Target-URI", ""), "lang_hint": header.get("WARC-Identified-Content-Language", ""),
                               "text": "".join(body).strip()}
                        n += 1
                        if limit and n >= limit:
                            return
                    header, body, in_body = {}, [], False
                elif not in_body:
                    if line.strip() == "":
                        in_body = True
                    elif ":" in line:
                        k, v = line.split(":", 1)
                        header[k.strip()] = v.strip()
                else:
                    body.append(line)
        except EOFError:                      # 文件还在下载 / 被截断：读到哪算哪
            pass
        if header.get("WARC-Type") == "conversion" and body:
            yield {"url": header.get("WARC-Target-URI", ""), "lang_hint": header.get("WARC-Identified-Content-Language", ""),
                   "text": "".join(body).strip()}


def write_jsonl(docs, path):
    with open(path, "w") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


EN_STOP = {"the", "of", "and", "to", "in", "a", "is", "that", "for", "it", "with", "as", "was", "on", "be", "by", "this", "are", "or", "at"}


def english_score(text):
    """最简单的语言识别：常见英文功能词占词数的比例。fastText 的 lid 模型做的是同一件事的高级版。"""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if len(words) < 20:
        return 0.0
    return sum(w in EN_STOP for w in words) / len(words)


def words_of(text):
    return re.findall(r"\S+", text)


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        print(f"  [{self.name}] {time.time() - self.t:.1f}s")


def fmt(n):
    return f"{n:,}"


def top_langs(docs, k=8):
    return Counter(d["lang_hint"].split(",")[0] or "?" for d in docs).most_common(k)
