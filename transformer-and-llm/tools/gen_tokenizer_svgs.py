"""预训练（02）「先讲明白」一章的三张图（手写 SVG，token 画成一格一格的方块）：

  pretrain-02-three-ways-to-cut.svg   同一句话按字符 / 按词 / 按子词切，各多少个 token
  pretrain-02-bpe-steps.svg           low / lower / newest / widest 语料上 BPE 的 8 次合并，每一步语料长什么样
  pretrain-02-tokenizers-side-by-side.svg  GPT-2 / cl100k / Qwen2.5 / DeepSeek-V3 切同一段英文、中文、代码

用法：python tools/gen_tokenizer_svgs.py [输出目录，默认 out/]
依赖：tiktoken、tokenizers（Qwen / DeepSeek 的 tokenizer.json 首次运行从 Hugging Face 下载）。
"""
import html
import os
import sys
from collections import Counter

import tiktoken
from tokenizers import Tokenizer

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")
os.makedirs(OUT, exist_ok=True)
FONT = "-apple-system,PingFang SC,Helvetica,Arial,sans-serif"
MONO = "SF Mono,Menlo,Consolas,monospace"
PALETTE = ["#dbe7f6", "#fde3c8", "#d9efdc", "#f6d6d6", "#e6def3", "#fbf0c4", "#d5eef2", "#ecdccf"]
STROKE = ["#3b6fb6", "#e0812c", "#4a9a5b", "#c94c4c", "#7b5ea7", "#c9a227", "#3a9aa6", "#a0724f"]


def tw(s, size=12, mono=False):
    """粗略的文字宽度估计：CJK 一个字一个 em，ASCII 半个多。"""
    w = 0
    for ch in s:
        if "\u3000" <= ch <= "\u9fff" or "\uff00" <= ch <= "\uffef":
            w += size * 1.0
        elif mono:
            w += size * 0.62
        else:
            w += size * (0.36 if ch in " il.,:;'|!" else 0.58)
    return w


def esc(s):
    return html.escape(s).replace(" ", "\u2423")   # 空格画成 ␣，不然看不见


class SVG:
    def __init__(self, w):
        self.w = w
        self.parts = []

    def text(self, x, y, s, size=12, anchor="start", weight="normal", fill="#222", mono=False):
        fam = f' font-family="{MONO}"' if mono else ""
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
                          f'font-weight="{weight}" fill="{fill}"{fam}>{html.escape(s)}</text>')

    def tokens(self, x, y, toks, size=12, h=20, gap=3, mono=False, maxw=None, color_by=None):
        """从 (x, y) 开始画一行 token 方块，返回结束 x。maxw 超出时换行（返回最后一行的 x 与用了几行）。"""
        cx, rows = x, 1
        for i, t in enumerate(toks):
            label = esc(t) if t.strip() or t == "" else "\u2423" * len(t)
            w = max(tw(label, size, mono) + 10, 16)
            if maxw and cx + w > x + maxw:
                cx = x; rows += 1; y += h + 6
            k = (color_by(t) if color_by else i) % len(PALETTE)
            self.parts.append(f'<rect x="{cx:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h}" rx="3" '
                              f'fill="{PALETTE[k]}" stroke="{STROKE[k]}" stroke-width="0.8"/>')
            fam = f' font-family="{MONO}"' if mono else ""
            self.parts.append(f'<text x="{cx + w / 2:.1f}" y="{y + h * 0.68:.1f}" font-size="{size}" '
                              f'text-anchor="middle" fill="#222"{fam}>{label}</text>')
            cx += w + gap
        return cx, rows

    def save(self, name, h):
        """h 是画完之后算出来的高度——SVG 必须带 width / height，否则在收缩包裹的图片容器里会被算成 0 × 0。"""
        head = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{h:.0f}" viewBox="0 0 {self.w} {h:.0f}" '
                f'font-family="{FONT}" font-size="12">', f'<rect width="{self.w}" height="{h:.0f}" fill="#fff"/>']
        path = os.path.join(OUT, name)
        open(path, "w").write("\n".join(head + self.parts + ["</svg>"]))
        print("  图已保存", path)


# ---------------------------------------------------------------- 图 1：三种切法
def fig_three_ways():
    sent = "Pretraining tokenizers is unglamorous work."
    enc = tiktoken.get_encoding("gpt2")
    sub = [enc.decode([t]) for t in enc.encode(sent)]
    words = ["Pretraining", " tokenizers", " is", " unglamorous", " work", "."]
    chars = list(sent)
    svg = SVG(740)
    svg.text(370, 22, f"同一句话：\u201c{sent}\u201d，三种切法", 14, "middle", "bold")
    rows = [("按字符切", chars, f"{len(chars)} 个 token；词表只要 26 个字母 + 标点，但序列很长，模型要自己学\u201ct-h-e 是一个词\u201d"),
            ("按词切", words, f"{len(words)} 个 token；词表要几十万，\u201cunglamorous\u201d\u201ctokenizers\u201d这种少见词不在词表里就成了 <unk>，信息全丢"),
            ("按子词切（GPT-2 的 BPE）", sub, f"{len(sub)} 个 token；常见词（is、work）整个一格，少见词拆成碎片（token+izers、un+g+lam+orous），任何字符串都能表示")]
    y = 48
    for title, toks, note in rows:
        svg.text(20, y + 14, title, 12.5, weight="bold")
        _, n = svg.tokens(20, y + 24, toks, size=11.5, h=20, maxw=700)
        y += 24 + n * 26
        svg.text(20, y + 10, note, 11, fill="#555")
        y += 30
    svg.save("pretrain-02-three-ways-to-cut.svg", y)


# ---------------------------------------------------------------- 图 2：BPE 的 8 次合并
def bpe_steps(corpus, n_merges):
    """最小 BPE：语料是 {词: 次数}，词以空格开头（与 GPT-2 一样把空格算进词里）。返回每一步 (合并对, 次数, 语料状态)。"""
    words = {tuple(" " + w): c for w, c in corpus.items()}
    states = [(None, None, dict(words))]
    for _ in range(n_merges):
        pairs = Counter()
        for w, c in words.items():
            for a, b in zip(w, w[1:]):
                pairs[(a, b)] += c
        (a, b), cnt = pairs.most_common(1)[0]
        new = {}
        for w, c in words.items():
            out, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                    out.append(a + b); i += 2
                else:
                    out.append(w[i]); i += 1
            new[tuple(out)] = c
        words = new
        states.append(((a, b), cnt, dict(words)))
    return states


def fig_bpe_steps():
    corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
    states = bpe_steps(corpus, 8)
    svg = SVG(800)
    svg.text(400, 20, "BPE 训练：每一步把语料里最常相邻的一对合并成一个新 token（语料只有四个词型，各出现 5 / 2 / 6 / 3 次）", 12.5, "middle", "bold")
    svg.text(20, 44, "第几步 · 合并了谁 · 相邻次数 = 含这一对的词的出现次数之和", 11, fill="#555")
    svg.text(310, 44, "合并之后四个词各切成什么样（␣ 是词前的空格；列头是每个词在语料里出现的次数）", 11, fill="#555")
    y = 60
    order = [" low", " lower", " newest", " widest"]
    x = 310
    for w in order:
        toks = next(list(ws) for ws in states[0][2] if "".join(ws) == w)
        x0 = x
        x, _ = svg.tokens(x, y, toks, size=11, h=19, gap=1.5, color_by=lambda t: 0)
        svg.text((x0 + x - 1.5) / 2, y + 36, f"{w.strip()} ×{corpus[w.strip()]}", 11, "middle", "bold", fill="#3b6fb6")
        x += 10
    y += 48
    for k, (pair, cnt, words) in enumerate(states):
        if k == 0:
            svg.text(20, y + 14, "初始：每个字符一个 token", 11.5, weight="bold")
        else:
            prev = states[k - 1][2]
            parts = [f"{c}" for w in order for ws, c in prev.items() if "".join(ws) == w and pair in zip(ws, ws[1:])]
            svg.text(20, y + 14, f"第 {k} 步：", 11.5, weight="bold")
            svg.tokens(70, y + 1, [pair[0], "+", pair[1], "→", pair[0] + pair[1]], size=11, h=19, gap=2,
                       color_by=lambda t, p=pair: 7 if t in ("+", "→") else (0 if t == p[0] else 1 if t == p[1] else 2))
            svg.text(298, y + 14, f"{cnt} 次 = {' + '.join(parts)}", 10.5, "end", fill="#555")
        x = 310
        for w in order:
            toks = next(list(ws) for ws in words if "".join(ws) == w)
            x, _ = svg.tokens(x, y + 1, toks, size=11, h=19, gap=1.5, color_by=lambda t: min(len(t) - 1, 5))
            x += 10
        y += 36
    svg.text(20, y + 8, "次数按词频加权：e+s 只在 newest、widest 两个词型里相邻，但两词共出现 6 + 3 = 9 次，比任何别的对都多，所以第 1 步合并它。", 10.5, fill="#555")
    svg.text(20, y + 24, "颜色 = token 的长度（1 个字符最浅）。8 步之后 low、est、␣new 各成了一个 token；训练的产物就是这 8 条合并规则，编码新词时按同样顺序再走一遍。", 10.5, fill="#555")
    svg.save("pretrain-02-bpe-steps.svg", y + 36)


# ---------------------------------------------------------------- 图 3：四个 tokenizer 并排
def fig_side_by_side():
    samples = [("英文", "The tokenizer decides how much every sentence costs."),
               ("中文", "分词器决定一句话变成多少个 token，也决定了它的价钱。"),
               ("Python", "        self.qkv = nn.Linear(d, 3 * d, bias=False)")]
    encs = []
    for name, key in [("GPT-2（50K）", "gpt2"), ("cl100k（100K，≈ Llama 3 的英文部分）", "cl100k_base")]:
        e = tiktoken.get_encoding(key)
        encs.append((name, lambda s, e=e: [e.decode_single_token_bytes(t) for t in e.encode(s)]))
    for name, repo in [("Qwen2.5（152K）", "Qwen/Qwen2.5-7B"), ("DeepSeek-V3（129K）", "deepseek-ai/DeepSeek-V3")]:
        t = Tokenizer.from_pretrained(repo)
        encs.append((name, lambda s, t=t: [t.decode([i], skip_special_tokens=False).encode() if t.decode([i]) else b"\xff"
                                            for i in t.encode(s, add_special_tokens=False).ids]))

    def show(bs):
        try:
            s = bs.decode()
            return s if s else "\ufffd"
        except UnicodeDecodeError:
            return "\ufffd"                                    # 不完整的 UTF-8 字节：一个汉字被切成了两三个 token

    svg = SVG(740)
    svg.text(370, 20, "同一段文字，四个 tokenizer 各切成几个 token（� 是不到一个字的字节碎片）", 13, "middle", "bold")
    y = 40
    for label, s in samples:
        svg.text(20, y + 12, f"{label}：{s.strip()}", 11.5, weight="bold", mono=(label == "Python"))
        y += 22
        for name, fn in encs:
            toks = fn(s)
            svg.text(20, y + 14, name, 10.5, fill="#444")
            x, n = svg.tokens(230, y + 1, [show(b) for b in toks], size=10.5, h=19, gap=1.5, mono=(label == "Python"), maxw=430)
            svg.text(725, y + 14, f"{len(toks)} 个", 11, "end", weight="bold")
            y += 26 * n + 2
        y += 16
    svg.text(20, y + 4, "英文四家几乎一样（词表 50K 时常用词就已经各是一个 token）；差别全在中文与代码：谁的训练语料里有，谁就省。", 10.5, fill="#555")
    svg.save("pretrain-02-tokenizers-side-by-side.svg", y + 14)


if __name__ == "__main__":
    fig_three_ways()
    fig_bpe_steps()
    fig_side_by_side()
