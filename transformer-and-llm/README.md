# Transformer 与 LLM：结构、实现与算量 — 配套代码

博客系列：[《Transformer 与 LLM：结构、实现与算量》](https://arganzheng.life/transformer-and-llm-for-infra-engineers.html)（十三篇：01–04 结构与实现、05–10 结构的演进、11–13 通用成本账）与紧接着的[《预训练：从 tokenizer 到训练配方》](https://arganzheng.life/pretraining-from-tokenizer-to-training-recipe.html)（六篇：01 端到端实跑、02–05 分词 / scaling law / 数据 / 配方、06 总结）。

## 预训练 01：一次预训练是怎么跑起来的（`pretrain_e2e/`）

[文章](https://arganzheng.life/pretraining-end-to-end-from-web-pages-to-a-model.html)在一台 MacBook 上把一次预训练从原始网页走到一个会续写英文的模型，八步各一个脚本，`run_all.sh` 一键跑完（MPS 上约一小时，纯 CPU 更久）：

| 文件 | 步 | 内容 | 依赖 |
|---|---|---|---|
| `step1_raw.py` | 1 原料 | 下载两个 Common Crawl WET 文件（约 215 MB，断点续传），读出 68,834 篇网页；长度与语言分布 | 无 |
| `step2_filter.py` | 2 过滤 | 语言识别（功能词比例）→ Gopher 文档级 → C4 行级 → Gopher 重复度，每道筛子的进出数与被删的例子 | 无 |
| `step3_dedup.py` | 3 去重 | 精确去重 + MinHash（128 哈希）/ LSH（16 × 8），候选对数、Jaccard 分布、近重复的例子 | NumPy |
| `step4_tokenizer.py` | 4 tokenizer | 在清洗后的语料上训 byte-level BPE（词表 4096）；样本切分；词表 264 → 16384 的压缩率扫描；与 GPT-2 对照 | tokenizers, tiktoken |
| `step5_pack.py` | 5 打包 | tokenize → `<\|endoftext\|>` → 按文档 98/2 切 → `train.bin` / `val.bin`（uint16）；文档长度与每条序列的边界数 | NumPy |
| `step6_isoflop.py` | 6 选尺寸 | 迷你 iso-FLOP：三个算力预算 × 八个模型尺寸，固定 C = 6ND 看 val loss（`--quick` 预算缩小 4 倍） | PyTorch |
| `step7_train.py` | 7 训练 | 按配方训一个小 GPT，一路记录七条曲线，画成一张图 | PyTorch |
| `step8_eval.py` | 8 评 | val loss → PPL → bits/byte；与 GPT-2 small 在同一批 val 文档上比 bits/byte；生成样本；试当助手用 | PyTorch, transformers（本地缓存 gpt2） |
| `domain_mixture.py` | 预训练 04 | 配比是取舍：网页 + Python 标准库两个域，代码占 0–100% 各训一个小模型，两个域各自的 val loss（MPS 约 12 分钟，`--quick`） | PyTorch |
| `show_before_after.py` | 预训练 04 | 一篇真实网页 C4 行级清理前后的对照 | 无 |
| `trainer.py` / `common.py` / `_plot.py` / `download.py` | — | 训练循环（记录七条曲线）、语料读写与语言识别、画图设置、断点续传下载 | — |

`data/` 下的下载文件与中间产物（约 1.8 GB）不入库；`expected/` 是每步的完整输出，`out/` 是文中的七张图。

## 第一段：结构与实现（01–04）

| 文件 | 文章 | 内容 | 依赖 |
|---|---|---|---|
| `attention_by_hand.py` | [01 Transformer 长什么样](https://arganzheng.life/transformer-architecture-from-a-sentence-to-the-next-token.html) | d=4、T=3 的 attention 六步手算与 `F.scaled_dot_product_attention` 对拍；换序实验；除 √d 的方差表 | PyTorch |
| `tools/gen_gpt2_attention_heatmap.py` | 同上 | GPT-2 small 两个真实 attention 头的热力图（SVG） | transformers（本地缓存 gpt2）, matplotlib |
| `tools/gen_attention_by_hand_svg.py` | 同上 | 六步手算图 | 无 |
| `position_two_methods.py` | 同上 | 两种给位置的方法用「猫追狗」/「狗追猫」算一遍：加位置向量 vs 转角度（RoPE），并画配图 | NumPy（配图需 matplotlib） |
| `token_journey.py` | [02 一个 token 的旅程](https://arganzheng.life/transformer-token-journey-training-and-inference.html) | 带 KV cache 的极小 GPT：训练侧形状、逐位置 loss、梯度、一步更新；prefill / decode；有无 cache 一致性与计时 | PyTorch |
| `tools/gen_token_journey_svg.py` | 同上 | 训练一步图、prefill / decode 图 | 无 |
| `nanogpt_model.py` | [03 nanoGPT model.py 逐行](https://arganzheng.life/nanogpt-model-py-line-by-line.html) | 原样 vendored 的 nanoGPT `model.py`（MIT） | PyTorch |
| `nanogpt_walkthrough.py` | 同上 | 极小配置的参数表与前向形状、GPT-2 small 参数量、与 HF GPT-2 对拍、`generate` 三种设置（`--no-hf` 跳过需要 transformers 的部分） | PyTorch, transformers |
| `nanogpt/` | [04 nanoGPT train.py 与实训](https://arganzheng.life/nanogpt-train-py-and-training-a-model-that-writes.html) | vendored 的 `train.py` / `configurator.py` / `sample.py` / `data/shakespeare_char/prepare.py`；`expected/train_shakespeare_char_{base,L2,L8}.txt` 是 MacBook（mps）上 2 / 4 / 8 层各 2000 步的日志 | PyTorch, numpy |
| `tools/gen_shakespeare_loss_svg.py` | 同上 | 三条 loss 曲线图 | 无 |

复现第四篇的实跑：

```bash
cd nanogpt && python data/shakespeare_char/prepare.py
python train.py --dataset=shakespeare_char --out_dir=out-shakespeare-char-base --device=mps --compile=False \
  --eval_interval=250 --eval_iters=20 --log_interval=50 --block_size=64 --batch_size=12 \
  --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
python sample.py --out_dir=out-shakespeare-char-base --device=mps --compile=False
```

## 第二、三段：结构的演进与成本账（05–13，原 01–08）

两个系列共用一个贯穿脚本 `llm_cost.py`：从第五篇的参数量开始，每篇加几个函数，前八版算完模型作为计算对象的成本表，第九到十二版（预训练系列 02–05）再算训练侧的账。文章里每一版都是独立可运行的完整代码，这里按篇保存为十二个文件，另加各篇的独立实验。文件名里的编号是脚本版本号（01–12），不是文章编号；多模态调整为第 10 篇后，脚本文件名与导入关系保持不变，下表按脚本版本排列。

| 文件 | 文章 | 内容 | 依赖 |
|---|---|---|---|
| `llm_cost_01_params.py` | [05 从 GPT-2 到 Llama——五处改动与参数量](https://arganzheng.life/transformer-anatomy-and-parameter-count.html) | 从超参数 / `config.json` 算逐组件参数量 | 无 |
| `tools/gen_gqa_mapping_svg.py` | 同上 | MHA / GQA / MQA 的 Q head 到 KV head 映射图；默认输出到 `out/`，也可传输出目录 | 无 |
| `llm_cost_02_flops_roofline.py` | [11 FLOPs、字节数与 Roofline](https://arganzheng.life/transformer-flops-bytes-and-roofline.html) | 每 token FLOPs、prefill / decode 的时间下界 | 无 |
| `llm_cost_03_attention_kv.py` | [06 Attention 变体与 KV cache](https://arganzheng.life/attention-variants-and-kv-cache.html) | MHA / GQA / MQA / MLA 的 KV 字节数与并发上限 | 无 |
| `llm_cost_04_long_context.py` | [07 位置编码与长上下文](https://arganzheng.life/positional-encoding-and-long-context.html) | 上下文长度扫描：KV、prefill、attention 占比 | 无 |
| `rope_numpy.py` | 同上 | RoPE 的 NumPy 实现、相对性验证、PI / NTK-aware / YaRN 波长表 | NumPy |
| `llm_cost_05_moe.py` | [08 MoE 的算量与通信](https://arganzheng.life/moe-compute-and-communication.html) | 总参数 / 激活参数、期望激活专家数、EP all-to-all 字节数 | 无 |
| `moe_layer_minimal.py` | 同上 | 最小 MoE 层：softmax 路由 + top-k + 共享专家 | PyTorch |
| `fp_formats.py` | [12 浮点格式与混合精度](https://arganzheng.life/floating-point-formats-and-mixed-precision.html) | 逐位构造 FP32 / FP16 / BF16 / FP8 | NumPy, PyTorch |
| `bf16_update_swallowed.py` | 同上 | BF16 权重更新被吃掉，为什么要 FP32 master | PyTorch |
| `gemm_error_vs_k.py` | 同上 | GEMM 误差随 k 的增长 | PyTorch |
| `llm_cost_06_dtype_state.py` | 同上 | dtype 字节表与训练状态显存 | 无 |
| `llm_cost_07_quant_specdec_lora.py` | [13 量化、投机解码与 LoRA](https://arganzheng.life/quantization-speculative-decoding-and-lora.html) | 量化字节数、投机解码加速比、LoRA 参数 | 无 |
| `llm_cost_08_multimodal.py` | [10 多模态成本](https://arganzheng.life/multimodal-vision-encoder-cost-and-image-token-kv.html) | vision encoder 参数与 FLOPs、image token 数与其在 decoder 的成本（复用第七版） | 无 |
| `vlm_cost_numbers.py` | 同上 | 文章里多模态各表的理论数字（BF16，H100 SXM） | 无 |
| `tools/gen_patch_merge_svg.py` | 同上 | 生成文中 patch → merge → token 的示意图 | 无 |
| `bpe_from_scratch.py` | [预训练 02 分词与词表](https://arganzheng.life/tokenizer-vocabulary-and-token-efficiency.html) | 从零实现 byte-level BPE；玩具例子；词表大小 → bytes/token 扫描（约 2 分钟，`--quick` 10 秒） | 无 |
| `tools/gen_tokenizer_svgs.py` | 同上 | 「先讲明白」一章的三张图：同一句话三种切法、BPE 八步合并、四个 tokenizer 并排切同一段中英文与代码 | tiktoken, tokenizers |
| `tokenizer_compare.py` | 同上 | GPT-2 / cl100k / o200k / Qwen2.5 / DeepSeek-V3 在英文、中文、代码、数字上的 token 效率 | tiktoken, tokenizers（联网下载词表） |
| `llm_cost_09_vocab.py` | 同上 | 词表参数与 lm_head 占比、logits 显存、每字符成本（复用第七版） | 无 |
| `scaling_law_fit.py` | [预训练 03 Scaling law](https://arganzheng.life/scaling-laws-and-compute-optimal-training.html) | CPU 上训 7 个字符级小模型，拟合 L(N) 并外推；常数 lr 下拟合 L(D)（约 10 分钟，`--quick` 1.5 分钟） | PyTorch |
| `llm_cost_10_scaling.py` | 同上 | Chinchilla 参数化与最优 N/D、真实模型的 D/N 与 GPU 小时、过训练代价、推理感知最优点、有效 token | 无 |
| `tools/paper_figures.py` | 同上（及预训练 04） | 从原论文 PDF 裁出教学引用的图：Kaplan 图 1、Chinchilla 图 1/2/3、FineWeb 管线图；PDF 缓存在 `data/papers/` | pymupdf |
| `tools/gen_scaling_svg.py` | 同上 | 文中的两栏图（实验拟合 + IsoFLOP 曲线） | 无 |
| `minhash_lsh.py` | [预训练 04 预训练数据工程](https://arganzheng.life/pretraining-data-pipeline-dedup-filtering-and-mixture.html) | MinHash + LSH 从零实现（FineWeb 配置），S 曲线验证，近重复文档演示 | 无 |
| `quality_filters.py` | 同上 | Gopher 文档级 / 重复度规则与 C4 行级规则，对典型网页文本逐条判定 | 无 |
| `llm_cost_11_data.py` | 同上 | 漏斗刻度、管线 CPU 小时、存储与训练读带宽、配比 → epoch | 无 |
| `recipe_intuition.py` | [预训练 05 训练配方与稳定性](https://arganzheng.life/pretraining-recipe-and-training-stability.html) | 「先讲明白」的四张图：学习率合适 / 太小 / 太大 / 太大且无 warmup 的曲线、大 lr 下 loss + attention logit + 梯度范数三条曲线同步看（有无 QK-norm）、z-loss、第一篇实训里那次真实小 spike 的放大（约 6 分钟，`--quick`） | PyTorch |
| `training_recipe_lab.py` | [预训练 05 训练配方与稳定性](https://arganzheng.life/pretraining-recipe-and-training-stability.html) | 四个子实验：`schedule` / `batch_lr` / `spike` / `zloss`（约 7 分钟，`--quick` 1 分钟） | PyTorch |
| `llm_cost_12_recipe.py` | 同上 | 公开配方的超参表与步数、DeepSeek 的 lr/batch 经验律、checkpoint 字节与写带宽、spike 回滚代价 | 无 |
| `tools/gen_schedule_svg.py` | 同上 | 文中的调度 / batch ramp 图 | 无 |

所有 `llm_cost_*` 都是纯 Python 标准库，直接 `python llm_cost_01_params.py` 即可；第一版还支持 `python llm_cost_01_params.py path/to/config.json` 读 transformers 风格的配置。第 09–12 版从第七版导入 `ModelConfig`，要在本目录下运行。`scaling_law_fit.py` 与 `training_recipe_lab.py` 用 PyTorch CPU，语料是 Python 自带的标准库源码，不需要下载；`tokenizer_compare.py` 是唯一需要联网的脚本（下载几个 tokenizer 的词表）。`expected/` 是每个脚本的完整输出。

这里的数字全部是理论下界或估算（参数量、FLOPs、字节数、峰值算力下的时间），不是实测；第 11 篇《前向的算量与访存量》讲了如何与实测对照。
