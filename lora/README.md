# LoRA 专题：配套实验

博客系列[《LoRA 专题：SFT 的默认微调方式》](https://arganzheng.life/lora-for-sft-from-low-rank-hypothesis-to-serving.html)（算法地图 L5 专题）三篇正文的全部数字都来自这里。

| 脚本 | 对应文章 | 子实验 |
|---|---|---|
| `01_low_rank.py` | [01 低秩假设](https://arganzheng.life/lora-low-rank-hypothesis-gradients-and-accounts.html) | `hand` 2×3 手算梯度 · `account` 四本账 · `speed` 一步耗时与算子数 · `init` 五种初始化 · `spectrum` 全量 ΔW 的谱与截秩 |
| `02_knobs.py` | [02 选参](https://arganzheng.life/lora-hyperparameters-rank-targets-alpha-lr-and-variants.html) | 全量 + 十二种 LoRA 配置各 80 步（`--list` 列出） |
| `03_deploy.py` | [03 工程](https://arganzheng.life/lora-in-production-adapters-merging-multi-lora-and-serving.html) | `files` · `merge` · `quant` · `multi` · `tokens` |
| `plot.py` | 01 / 02 | 从 `out/*.json` 画文章里的 SVG |

## 运行

```bash
pip install torch transformers peft trl datasets bitsandbytes kernels matplotlib   # 版本见 expected/ 开头
python 01_low_rank.py hand account            # 几秒，不加载大模型
python 02_knobs.py --quick                    # 每种配置 8 步，十分钟，确认环境
python 02_knobs.py                            # 全矩阵：8 线程 CPU 约 2 小时，MPS / CUDA 快 3–10 倍
python 01_low_rank.py speed init spectrum     # spectrum 读 02 存下的 out/ckpt/full_linear.pt
python 03_deploy.py                           # 读 02 存下的 out/ckpt/r16_all 与 r16_attn
python plot.py
```

- 模型 `Qwen/Qwen2.5-0.5B`（base），数据 `HuggingFaceH4/no_robots`、`Salesforce/wikitext`，首次运行会下载到 Hugging Face 缓存。
- 与 `../post-training/01_sft.py` 同一套配方（800 条、80 步、batch 4 × 512、只对回复算 loss）；训练前的两个基线数（验证回复 loss 2.4936、普通文本 loss 2.8748）与全量微调的结果（2.3924 / +0.0213）在 CPU 与 Apple Silicon 上一致。
- QLoRA（`r16_qlora`、`03_deploy.py quant`）需要 bitsandbytes；CPU 上还要 `pip install kernels` 并联网一次下载 `kernels-community/quantization-bitsandbytes`。没有就自动跳过。
- `out/` 里的 checkpoint（全量 2 GB、adapter 17 MB）与 JSON 不入库；`expected/` 是作者机器上的完整输出。
