# 后训练：从 SFT 到可验证奖励 — 配套实验

博客系列：[总纲](https://arganzheng.life/post-training-from-sft-to-verifiable-rewards.html)。八篇文章各配一个脚本，用 `trl` 在 Qwen2.5 的 0.5B / 1.5B 模型上把一条后训练流水线跑通：SFT → 奖励模型 → GRPO / DPO → RLVR → Agent RL → 蒸馏 → 评测。文中引用的数字都由这些脚本跑出。

| 脚本 | 文章 | 设备 | 完整运行 |
|---|---|---|---|
| `01_sft.py` | [SFT：指令数据、chat template、loss mask 与参数高效微调](https://arganzheng.life/sft-data-chat-template-loss-mask-and-peft.html) | MPS / CUDA（CPU 可 `--quick`） | 约 60 分钟 |

## 运行

```bash
pip install -r ../requirements.txt            # torch + transformers + trl + peft + datasets + accelerate
python 01_sft.py template padding             # 不训练的实验，几秒
python 01_sft.py --quick                      # 训练类实验缩到十几步，约 5 分钟
python 01_sft.py                              # 完整运行
```

约定：

- 模型与数据集（Qwen/Qwen2.5-0.5B、HuggingFaceH4/no_robots、Salesforce/wikitext）第一次运行时从 Hugging Face 下载到 `~/.cache/huggingface`（约 1.5 GB）；之后建议加 `HF_HUB_OFFLINE=1` 运行——未登录的 Hub 请求偶尔会卡住几分钟。
- 设备自动选择 `cuda` → `mps` → `cpu`（`ptlab.device()`）。数字来自作者的 Apple Silicon 笔记本（MPS，FP32）；CUDA 上会快得多，CPU 上慢 5–10 倍。
- 训练类实验支持 `--quick` 与按名字选子实验，`-h` 列出。
- `ptlab.py` 是各篇共用的小工具：加载模型、只在回复 token 上算 loss（`completion_loss`）、普通文本上算 loss（`text_loss`，衡量遗忘）、参数与训练状态的账。
- `expected/` 是作者机器上的完整输出。随机性、库版本与设备不同会让小数位不同，趋势应一致。
