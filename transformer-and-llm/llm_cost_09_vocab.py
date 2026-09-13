"""llm_cost.py 第九版（Transformer 与 LLM 09）：词表大小的账——embedding / lm_head 的参数占比、
lm_head 的 FLOPs 与字节数、logits 显存，以及"每个字符的成本"。ModelConfig / param_count 沿用第七版。
https://arganzheng.life/tokenizer-vocabulary-and-token-efficiency.html
"""
from dataclasses import replace
from llm_cost_07_quant_specdec_lora import (ModelConfig, LLAMA3_8B, LLAMA3_70B, H100,
                                            param_count, forward_flops_per_token, kv_bytes_per_token)

LLAMA2_7B = ModelConfig("Llama-2-7B", 4096, 32, 32, 32, 128, 11008, 32000)
QWEN25_7B = ModelConfig("Qwen2.5-7B", 3584, 28, 28, 4, 128, 18944, 152064)
QWEN25_05B = ModelConfig("Qwen2.5-0.5B", 896, 24, 14, 2, 64, 4864, 151936, tie_embeddings=True)
GEMMA2_2B = ModelConfig("Gemma-2-2B", 2304, 26, 8, 4, 256, 9216, 256000, tie_embeddings=True)


def vocab_account(cfg):
    p = param_count(cfg)
    vocab_params = p["embedding"] + p["lm_head"]
    lm_head_flops = 2 * cfg.vocab * cfg.hidden
    return {
        "total": p["total"],
        "vocab_params": vocab_params,
        "vocab_share": vocab_params / p["total"],
        "lm_head_flops": lm_head_flops,
        "lm_head_share": lm_head_flops / forward_flops_per_token(cfg),
        "lm_head_bytes": cfg.vocab * cfg.hidden * 2,            # BF16，decode 每步读一遍
    }


def logits_bytes(cfg, tokens, dtype_bytes=4):
    """训练时一个 micro-batch 的 logits：tokens × V × 4 B（FP32 算 CE），不分块就要一次放下。"""
    return tokens * cfg.vocab * dtype_bytes


def per_char_cost(cfg, chars_per_token):
    """把每 token 的 FLOPs 与 KV 换算到每个字符：tokenizer 的压缩率直接除进去。"""
    return {
        "flops_per_char": forward_flops_per_token(cfg) / chars_per_token,
        "kv_per_char": kv_bytes_per_token(cfg) / chars_per_token,
    }


def fmt(x):
    for unit, s in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= unit:
            return f"{x / unit:.2f}{s}"
    return f"{x:.0f}"


def main():
    print("=== 词表在参数、FLOPs、字节里的份额 ===")
    print(f"{'model':<14}{'V':>8}{'d':>6}{'参数':>9}{'词表参数':>10}{'占比':>7}{'lm_head FLOPs/tok':>18}{'占比':>7}{'lm_head 字节':>13}")
    for cfg in [LLAMA2_7B, LLAMA3_8B, LLAMA3_70B, QWEN25_7B, QWEN25_05B, GEMMA2_2B]:
        a = vocab_account(cfg)
        tie = " (tied)" if cfg.tie_embeddings else ""
        print(f"{cfg.name + tie:<14}{cfg.vocab:>8}{cfg.hidden:>6}{fmt(a['total']):>9}{fmt(a['vocab_params']):>10}"
              f"{a['vocab_share']:>7.1%}{fmt(a['lm_head_flops']):>18}{a['lm_head_share']:>7.1%}{fmt(a['lm_head_bytes']) + 'B':>13}")

    print("\n=== 同一个 8B 骨架换词表：V 从 32K 到 256K ===")
    print(f"{'V':>8}{'参数':>9}{'词表占比':>9}{'FLOPs/tok':>11}{'lm_head 占比':>13}{'decode 读 lm_head':>18}{'8K 序列 logits(FP32)':>21}")
    for v in [32000, 64000, 128256, 256000]:
        cfg = replace(LLAMA3_8B, vocab=v)
        a = vocab_account(cfg)
        t = a["lm_head_bytes"] / H100.bandwidth
        print(f"{v:>8}{fmt(a['total']):>9}{a['vocab_share']:>9.1%}{fmt(forward_flops_per_token(cfg)):>11}"
              f"{a['lm_head_share']:>13.1%}{t * 1e3:>15.2f} ms{logits_bytes(cfg, 8192) / 2**30:>17.1f} GiB")

    print("\n=== 每个字符的成本：Llama 2 → Llama 3 的 tokenizer（英文 3.17 → 3.94 字符/token，Llama 3 论文）===")
    rows = [("Llama-3-8B 骨架 + 32K 词表", replace(LLAMA3_8B, vocab=32000), 3.17),
            ("Llama-3-8B（128K 词表）", LLAMA3_8B, 3.94)]
    base = None
    for name, cfg, cpt in rows:
        c = per_char_cost(cfg, cpt)
        rel = "" if base is None else f"   ({c['flops_per_char'] / base['flops_per_char'] - 1:+.0%} FLOPs, {c['kv_per_char'] / base['kv_per_char'] - 1:+.0%} KV)"
        base = base or c
        print(f"  {name:<28} {fmt(forward_flops_per_token(cfg)):>8} FLOPs/tok  {cpt:.2f} 字符/tok  ->  "
              f"{fmt(c['flops_per_char'])} FLOPs/字符  {c['kv_per_char'] / 1024:.1f} KiB KV/字符{rel}")

    print("\n=== 中文：同一段话在不同 tokenizer 下的每字符成本（用 tokenizer_compare.py 测得的 token/汉字）===")
    for name, tok_per_han in [("cl100k（≈Llama 3）", 1.46), ("Qwen2.5", 0.79), ("DeepSeek-V3", 0.69)]:
        c = per_char_cost(LLAMA3_8B, 1 / tok_per_han)
        print(f"  {name:<18} {tok_per_han:.2f} token/汉字 -> 8B 规格每个汉字 {fmt(c['flops_per_char'])} FLOPs，{c['kv_per_char'] / 1024:.0f} KiB KV")


if __name__ == "__main__":
    main()
