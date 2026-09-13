"""llm_cost.py 第十一版（Transformer 与 LLM 11）：预训练数据的账——从 Common Crawl 到 15T token 的漏斗、
去重 / 过滤 / tokenize 的 CPU 小时、存储与训练时的读取带宽、数据配比换算成每个来源的 epoch 数、LSH 的 S 曲线。
https://arganzheng.life/pretraining-data-pipeline-dedup-filtering-and-mixture.html
"""

# ---- 公开数据集给出的漏斗刻度（token 数，量级）----
FUNNEL = [
    ("Common Crawl 全部快照的原始网页（WARC）", "约 30+ PB 压缩",           None),
    ("抽出正文后的文本（DCLM-Pool，2023 前全部快照）", "240T token",           240e12),
    ("URL / 语言 / 启发式过滤 + 去重后（FineWeb，96 个快照）", "15T token，44 TB parquet", 15e12),
    ("再按'教育价值'分类器打分 >= 2（FineWeb-Edu）", "5.4T token",           5.4e12),
    ("分类器打分 >= 3（FineWeb-Edu 高分子集）", "1.3T token",                1.3e12),
    ("DCLM-baseline：启发式 + 去重 + fastText 取前 10%", "3.8T token",         3.8e12),
]

# ---- 各步骤的单核吞吐假设（量级，取自公开工具的常见数字；换机器要重测）----
DOCS_PER_TOKEN = 1 / 1000              # 平均每篇网页正文约 1000 token
EXTRACT_DOCS_PER_CORE_S = 100          # trafilatura 一类正文抽取：每核每秒约 100 页
TOKENIZE_TOK_PER_CORE_S = 250_000      # Hugging Face tokenizers：每核每秒约 25 万 token
MINHASH_OPS_PER_CORE_S = 1e8           # 每核每秒约 1e8 次 (a·x+b) mod p
N_HASH = 112


def cpu_hours_extract(pages):
    return pages / EXTRACT_DOCS_PER_CORE_S / 3600


def cpu_hours_tokenize(tokens):
    return tokens / TOKENIZE_TOK_PER_CORE_S / 3600


def cpu_hours_minhash(tokens, n_hash=N_HASH):
    """每篇文档的每个 5-gram 要过 n_hash 个哈希；5-gram 数 ≈ 词数 ≈ token 数 × 0.75。"""
    return tokens * 0.75 * n_hash / MINHASH_OPS_PER_CORE_S / 3600


def storage_bytes(tokens, bytes_per_token=4):
    """token id 以 uint32 存（128K 词表放不进 uint16）。"""
    return tokens * bytes_per_token


def read_bandwidth(tokens, days):
    """训练时从存储读 token 的平均带宽。"""
    return tokens * 4 / (days * 86400)


def mixture_epochs(total_tokens, mix):
    """mix: {来源: (占比, 唯一 token 数)} → 每个来源要跑多少 epoch。"""
    return {k: (w * total_tokens, w * total_tokens / u) for k, (w, u) in mix.items()}


def p_candidate(j, bands=14, rows=8):
    return 1 - (1 - j ** rows) ** bands


def fmt(x):
    for unit, s in ((1e15, "P"), (1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= unit:
            return f"{x / unit:.3g}{s}"
    return f"{x:.3g}"


def main():
    print("=== 漏斗：从网页到训练 token（公开数据集的刻度）===")
    base = FUNNEL[1][2]
    for name, label, tok in FUNNEL:
        ratio = f"{tok / base:>7.1%}" if tok else "       "
        print(f"  {ratio}  {label:<26} {name}")
    print("  15T 的通用网页语料是抽出正文后文本的 6%；模型打分后的'高质量'子集只剩 0.5%–1.6%。")

    print("\n=== 管线的 CPU 账（15T token 的最终语料，假设见源码顶部）===")
    final_tokens = 15e12
    pre_dedup_tokens = 30e12           # 去重前约为最终的 2 倍（RedPajama-v2：30T → 20T）
    pages = 250e9                      # 96 个快照合计约 2500 亿次抓取（跨快照大量重复）
    rows = [
        ("正文抽取（所有抓取页）", cpu_hours_extract(pages)),
        ("MinHash 签名（去重前文本）", cpu_hours_minhash(pre_dedup_tokens)),
        ("tokenize（最终语料）", cpu_hours_tokenize(final_tokens)),
    ]
    for name, h in rows:
        print(f"  {name:<24} {h:>12,.0f} 核·小时 = {h / 1000 / 24:>6.1f} 天 @ 1000 核")
    print("  对照：Llama-3 8B 的训练是 146 万 GPU 小时；数据管线的成本以 CPU 核·小时计、贵在正文抽取与去重，不在 tokenize。")

    print("\n=== 存储与训练时的读取带宽 ===")
    print(f"  15T token 的 id（uint32）      {storage_bytes(15e12) / 1e12:>6.0f} TB   （FineWeb 的文本 parquet 44 TB；uint16 放不下 128K 词表）")
    print(f"  Llama-3 405B：15.6T token / 78 天  平均 {read_bandwidth(15.6e12, 78) / 1e6:>5.0f} MB/s   （3084 万 GPU 小时 ÷ 16K 张 H100 ≈ 78 天；读数据的带宽只相当于一块硬盘）")
    print(f"  Llama-3 8B：15T token / 15 天（假设）  平均 {read_bandwidth(15e12, 15) / 1e6:>5.0f} MB/s")
    print("  训练时的数据 I/O 微不足道；成本全在训练之前的离线管线里。")

    print("\n=== 配比换算成 epoch：Llama 3 的 15T = 50% 通用 + 25% 数学推理 + 17% 代码 + 8% 多语言 ===")
    mix = {  # (占比, 可得的唯一 token 数——后者是公开数据集给出的量级，用作估计)
        "通用网页（FineWeb 级）": (0.50, 15e12),
        "数学与推理（网页中筛出 + 论文 + 教材）": (0.25, 0.5e12),
        "代码（The Stack v2 去重后）": (0.17, 0.9e12),
        "多语言": (0.08, 3e12),
    }
    print(f"  {'来源':<32}{'目标 token':>10}{'唯一 token(估)':>15}{'epoch':>7}  Muennighoff 折算")
    from math import exp
    for k, (tok, ep) in mixture_epochs(15e12, mix).items():
        u = mix[k][1]
        eff = u + u * 15.39 * (1 - exp(-(ep - 1) / 15.39)) if ep > 1 else tok
        print(f"  {k:<32}{fmt(tok):>10}{fmt(u):>15}{ep:>7.1f}  有效 {fmt(eff)}（{eff / tok:.0%}）")
    print("  25% 数学推理意味着有限的数学语料要跑多个 epoch，或者靠合成 / 改写扩充；这是 2024 年后合成数据兴起的算术原因。")

    print("\n=== MinHash LSH 的 S 曲线：14 桶 × 8 行 ===")
    print("  " + "  ".join(f"J={j:.2f}:{p_candidate(j):.3f}" for j in [0.5, 0.6, 0.7, 0.72, 0.8, 0.9]))
    print("  阈值 (1/14)^(1/8) = 0.72：Jaccard 0.6 的文档对只有 21% 会被比较，0.8 的有 92%。")


if __name__ == "__main__":
    main()
