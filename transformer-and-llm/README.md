# Transformer 与 LLM：结构、算量与数值 — 配套代码

博客系列：[《Transformer 与 LLM：结构、算量与数值》](https://arganzheng.life/transformer-and-llm-for-infra-engineers.html)（八篇，成本表）与紧接着的[《预训练：从 tokenizer 到训练配方》](https://arganzheng.life/pretraining-from-tokenizer-to-training-recipe.html)（四篇，训练侧）。两个系列共用一个贯穿脚本 `llm_cost.py`：从第一篇的参数量开始，每篇加几个函数，前八版算完模型作为计算对象的成本表，第九到十二版（预训练系列 01–04）再算训练侧的账。文章里每一版都是独立可运行的完整代码，这里按篇保存为十二个文件，另加各篇的独立实验。

| 文件 | 文章 | 内容 | 依赖 |
|---|---|---|---|
| `llm_cost_01_params.py` | [01 结构与参数量](https://arganzheng.life/transformer-anatomy-and-parameter-count.html) | 从超参数 / `config.json` 算逐组件参数量 | 无 |
| `llm_cost_02_flops_roofline.py` | [02 FLOPs、字节数与 Roofline](https://arganzheng.life/transformer-flops-bytes-and-roofline.html) | 每 token FLOPs、prefill / decode 的时间下界 | 无 |
| `llm_cost_03_attention_kv.py` | [03 Attention 变体与 KV cache](https://arganzheng.life/attention-variants-and-kv-cache.html) | MHA / GQA / MQA / MLA 的 KV 字节数与并发上限 | 无 |
| `llm_cost_04_long_context.py` | [04 位置编码与长上下文](https://arganzheng.life/positional-encoding-and-long-context.html) | 上下文长度扫描：KV、prefill、attention 占比 | 无 |
| `rope_numpy.py` | 同上 | RoPE 的 NumPy 实现、相对性验证、PI / NTK-aware / YaRN 波长表 | NumPy |
| `llm_cost_05_moe.py` | [05 MoE 的算量与通信](https://arganzheng.life/moe-compute-and-communication.html) | 总参数 / 激活参数、期望激活专家数、EP all-to-all 字节数 | 无 |
| `moe_layer_minimal.py` | 同上 | 最小 MoE 层：softmax 路由 + top-k + 共享专家 | PyTorch |
| `fp_formats.py` | [06 浮点格式与混合精度](https://arganzheng.life/floating-point-formats-and-mixed-precision.html) | 逐位构造 FP32 / FP16 / BF16 / FP8 | NumPy, PyTorch |
| `bf16_update_swallowed.py` | 同上 | BF16 权重更新被吃掉，为什么要 FP32 master | PyTorch |
| `gemm_error_vs_k.py` | 同上 | GEMM 误差随 k 的增长 | PyTorch |
| `llm_cost_06_dtype_state.py` | 同上 | dtype 字节表与训练状态显存 | 无 |
| `llm_cost_07_quant_specdec_lora.py` | [07 量化、投机解码与 LoRA](https://arganzheng.life/quantization-speculative-decoding-and-lora.html) | 量化字节数、投机解码加速比、LoRA 参数 | 无 |
| `llm_cost_08_multimodal.py` | [08 多模态成本](https://arganzheng.life/multimodal-vision-encoder-cost-and-image-token-kv.html) | vision encoder 参数与 FLOPs、image token 数与其在 decoder 的成本（复用第七版） | 无 |
| `vlm_cost_numbers.py` | 同上 | 文章里多模态各表的理论数字（BF16，H100 SXM） | 无 |
| `tools/gen_patch_merge_svg.py` | 同上 | 生成文中 patch → merge → token 的示意图 | 无 |
| `bpe_from_scratch.py` | [预训练 01 分词与词表](https://arganzheng.life/tokenizer-vocabulary-and-token-efficiency.html) | 从零实现 byte-level BPE；玩具例子；词表大小 → bytes/token 扫描（约 2 分钟，`--quick` 10 秒） | 无 |
| `tokenizer_compare.py` | 同上 | GPT-2 / cl100k / o200k / Qwen2.5 / DeepSeek-V3 在英文、中文、代码、数字上的 token 效率 | tiktoken, tokenizers（联网下载词表） |
| `llm_cost_09_vocab.py` | 同上 | 词表参数与 lm_head 占比、logits 显存、每字符成本（复用第七版） | 无 |
| `scaling_law_fit.py` | [预训练 02 Scaling law](https://arganzheng.life/scaling-laws-and-compute-optimal-training.html) | CPU 上训 7 个字符级小模型，拟合 L(N) 并外推；常数 lr 下拟合 L(D)（约 10 分钟，`--quick` 1.5 分钟） | PyTorch |
| `llm_cost_10_scaling.py` | 同上 | Chinchilla 参数化与最优 N/D、真实模型的 D/N 与 GPU 小时、过训练代价、推理感知最优点、有效 token | 无 |
| `tools/gen_scaling_svg.py` | 同上 | 文中的两栏图（实验拟合 + IsoFLOP 曲线） | 无 |
| `minhash_lsh.py` | [预训练 03 预训练数据工程](https://arganzheng.life/pretraining-data-pipeline-dedup-filtering-and-mixture.html) | MinHash + LSH 从零实现（FineWeb 配置），S 曲线验证，近重复文档演示 | 无 |
| `quality_filters.py` | 同上 | Gopher 文档级 / 重复度规则与 C4 行级规则，对典型网页文本逐条判定 | 无 |
| `llm_cost_11_data.py` | 同上 | 漏斗刻度、管线 CPU 小时、存储与训练读带宽、配比 → epoch | 无 |
| `training_recipe_lab.py` | [预训练 04 训练配方与稳定性](https://arganzheng.life/pretraining-recipe-and-training-stability.html) | 四个子实验：`schedule` / `batch_lr` / `spike` / `zloss`（约 7 分钟，`--quick` 1 分钟） | PyTorch |
| `llm_cost_12_recipe.py` | 同上 | 公开配方的超参表与步数、DeepSeek 的 lr/batch 经验律、checkpoint 字节与写带宽、spike 回滚代价 | 无 |
| `tools/gen_schedule_svg.py` | 同上 | 文中的调度 / batch ramp 图 | 无 |

所有 `llm_cost_*` 都是纯 Python 标准库，直接 `python llm_cost_01_params.py` 即可；第一版还支持 `python llm_cost_01_params.py path/to/config.json` 读 transformers 风格的配置。第 09–12 版从第七版导入 `ModelConfig`，要在本目录下运行。`scaling_law_fit.py` 与 `training_recipe_lab.py` 用 PyTorch CPU，语料是 Python 自带的标准库源码，不需要下载；`tokenizer_compare.py` 是唯一需要联网的脚本（下载几个 tokenizer 的词表）。`expected/` 是每个脚本的完整输出。

这里的数字全部是理论下界或估算（参数量、FLOPs、字节数、峰值算力下的时间），不是实测；文章第二篇讲了如何与实测对照。
