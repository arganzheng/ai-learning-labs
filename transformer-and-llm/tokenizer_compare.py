"""真实 tokenizer 对比（Transformer 与 LLM 09）：GPT-2 / cl100k / o200k / Qwen2.5 / DeepSeek-V3 在英文、中文、代码、数字上的 token 效率。
https://arganzheng.life/tokenizer-vocabulary-and-token-efficiency.html

依赖：pip install tiktoken tokenizers（后者首次运行会从 Hugging Face 下载 tokenizer.json，几 MB）。
Llama 3 的 tokenizer 需要登录才能下载，这里用同为 tiktoken 系、词表 100K 的 cl100k_base 近似
（Llama 3 的 128K 词表 = cl100k 的 100K + 28K 非英语 token）。
"""
import sys

try:
    import tiktoken
    from tokenizers import Tokenizer
except ImportError:
    sys.exit("需要 tiktoken 与 tokenizers：pip install tiktoken tokenizers")

SAMPLES = {
    "英文": "The tokenizer decides how many tokens a sentence becomes, and therefore how much every "
            "sentence costs: the model pays the same price for every token regardless of its length.",
    "中文": "分词器决定一句话变成多少个 token，也就决定了这句话的成本：无论 token 长短，模型为每个 token 付同样的价钱。",
    "Python": "class Attention(nn.Module):\n    def __init__(self, d, n_heads):\n        super().__init__()\n"
              "        self.qkv = nn.Linear(d, 3 * d, bias=False)\n        self.out = nn.Linear(d, d, bias=False)\n",
    "数字": "In 2024 the model was trained on 15000000000000 tokens at a cost of 3.8e25 FLOPs, or 1234567.89 dollars per step.",
}


class Enc:
    def __init__(self, name, fn, vocab):
        self.name, self.fn, self.vocab = name, fn, vocab


def load():
    encs = []
    for name in ["gpt2", "cl100k_base", "o200k_base"]:
        e = tiktoken.get_encoding(name)
        encs.append(Enc(name, lambda s, e=e: [e.decode([t]) for t in e.encode(s)], e.n_vocab))
    for repo, short in [("Qwen/Qwen2.5-7B", "Qwen2.5"), ("deepseek-ai/DeepSeek-V3", "DeepSeek-V3")]:
        try:
            t = Tokenizer.from_pretrained(repo)
        except Exception as ex:  # 无网络时跳过
            print(f"[跳过 {short}: {ex.__class__.__name__}]")
            continue
        encs.append(Enc(short, lambda s, t=t: [t.decode([i]) for i in t.encode(s, add_special_tokens=False).ids],
                        t.get_vocab_size()))
    return encs


def main():
    encs = load()
    print("=== 每个样本的 token 数 / 每 token 字符数（越大越省）===")
    print(f"{'tokenizer':<14}{'词表':>9} " + "".join(f"{k:>18}" for k in SAMPLES))
    for e in encs:
        cells = []
        for s in SAMPLES.values():
            n = len(e.fn(s))
            cells.append(f"{n:>4} tok {len(s) / n:>5.2f} c/t")
        print(f"{e.name:<14}{e.vocab:>9} " + "".join(f"{c:>18}" for c in cells))

    print("\n=== 中文：每个汉字平均多少 token ===")
    zh = SAMPLES["中文"]
    han = sum("\u4e00" <= ch <= "\u9fff" for ch in zh)
    for e in encs:
        toks = e.fn(zh)
        print(f"  {e.name:<14} {len(toks) / han:.2f} token/汉字   例: {toks[:8]}")

    print("\n=== 数字：同一个 13 位数被切成什么 ===")
    for e in encs:
        print(f"  {e.name:<14} {e.fn('15000000000000')}")

    print("\n=== 缩进：8 个空格是 1 个 token 还是 8 个 ===")
    for e in encs:
        print(f"  {e.name:<14} {len(e.fn(' ' * 8 + 'return x'))} token  {e.fn(' ' * 8 + 'return x')}")


if __name__ == "__main__":
    main()
