# Transformer 与 LLM：结构、算量与数值 — 配套代码

博客系列：[总纲](https://arganzheng.life/transformer-and-llm-for-infra-engineers.html)。这个系列的贯穿脚本是 `llm_cost.py`：从第一篇的参数量开始，每篇加几个函数，到第八篇算完多模态。文章里每一版都是独立可运行的完整代码，这里按篇保存为八个文件，另加几篇文章里的独立实验。

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

所有 `llm_cost_*` 都是纯 Python 标准库，直接 `python llm_cost_01_params.py` 即可；第一版还支持 `python llm_cost_01_params.py path/to/config.json` 读 transformers 风格的配置。`expected/` 是每个脚本的完整输出。

这里的数字全部是理论下界或估算（参数量、FLOPs、字节数、峰值算力下的时间），不是实测；文章第二篇讲了如何与实测对照。
