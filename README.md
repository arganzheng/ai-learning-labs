# ai-learning-labs

[arganzheng.life](https://arganzheng.life) 上 AI 系列文章的配套代码：文中引用的每个数字、每段输出都由这里的脚本跑出来。目的有两个——让读者能下载、运行、对照、改着玩；也让作者写文章时用过的验证代码不至于丢掉。

博客分三张学习地图（[AI-Infra 工程师](https://arganzheng.life/ai-infra-learning-roadmap.html)、[AI 算法工程师](https://arganzheng.life/ai-algorithm-engineer-learning-roadmap.html)、[AI 应用工程师](https://arganzheng.life/ai-application-engineer-learning-roadmap.html)），地图由若干**系列**组成，一个系列可以同时出现在两张地图里。代码仓按系列分目录，目录名就是博客里的系列 key；下表是地图 → 层 → 目录的索引。

| 目录 | 系列 | 所在地图与层 | 依赖 |
|---|---|---|---|
| [`python-for-ai-infra/`](python-for-ai-infra/) | Python 在 AI-Infra：从语言机制到生产交付 | Infra L1 | Python 3.10+ 标准库 |
| [`cpp-for-ai-infra/`](cpp-for-ai-infra/) | C++ 在 AI-Infra：从对象模型到算子扩展 | Infra L1 | C++17 编译器 + make |
| [`transformer-and-llm/`](transformer-and-llm/) | Transformer 与 LLM：结构、算量与数值（8 篇）+ 预训练：从 tokenizer 到训练配方（4 篇，`llm_cost_09`–`_12` 及配套实验） | Infra L2 · 算法 L4（成本表两张地图共享；预训练只在算法地图） | 纯 Python；部分实验 NumPy / PyTorch；tokenizer 对比需 tiktoken + tokenizers |
| [`algorithm-tooling/`](algorithm-tooling/) | 算法工程师的工具箱：从一个想法到一次能跑的实验 | 算法 L1 | numpy、torch（CPU）、pandas、matplotlib；04 需 transformers / peft / trl + 下载 Qwen2.5-0.5B |
| [`classical-ml/`](classical-ml/) | LLM 时代的经典机器学习：只讲它在哪里重现 | 算法 L2 | numpy、scikit-learn、matplotlib |
| [`deep-learning-foundations/`](deep-learning-foundations/) | 深度学习基础：从反向传播到残差 | 算法 L3 | NumPy；05 / 06 需 PyTorch（CPU） |
| [`post-training/`](post-training/) | 后训练：从 SFT 到可验证奖励 | 算法 L5 | PyTorch + transformers / trl / peft；MPS 或 CUDA |
| [`rl-post-training-infra/`](rl-post-training-infra/) | RL 后训练基础设施：rollout 与训练如何共享一组 GPU | Infra L4（09） | 纯 Python（账本）；后续实验需 verl + 8 卡 |
| [`diffusion-inference-infra/`](diffusion-inference-infra/) | 扩散模型推理基础设施：从一次去噪到一个生成服务 | Infra L4（10） | 纯 Python（账本） |
| [`coding-interview/`](coding-interview/) | 面试手撕代码：从 LeetCode 中等题到 Transformer 组件（19 篇） | 独立系列，不属于三张地图 | `python/` 标准库；`java/` JDK 21；`ai/` numpy + torch（CPU）；`infra/` C++17 + make |

尚未收录代码的系列（PyTorch 深度实践、GPU Kernel 工程、通信与互联、大规模训练、vLLM、AI 平台工程、开源贡献、算法地图 L0 数学、应用地图）会在有可运行示例时按同样方式加目录。

## 使用

```bash
git clone https://github.com/arganzheng/ai-learning-labs.git
cd ai-learning-labs
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy + torch + torchvision + pandas + matplotlib + scikit-learn，CPU 即可
```

然后进到某个系列目录，按它的 README 运行。约定：

- 每个脚本头部写着对应文章的链接与它验证的结论；
- 训练类脚本支持 `--quick`（一分钟内跑完，用来确认环境）和按名字选子实验，`-h` 列出；
- 每个系列的 `expected/` 是作者机器上（Apple Silicon 笔记本，8 线程，PyTorch CPU）的完整输出。不同 CPU、BLAS、线程数、PyTorch 版本下小数末位、耗时、偶尔的随机波动会不同，趋势和结论应一致；如果结论不一致，欢迎开 issue。
- 数据集（MNIST）在首次运行时自动下载到系列目录下的 `data/`，不进仓库。

## 与文章的关系

文章是主体，代码是附件：文中会把推导、数字与结论讲完，代码只负责"你可以自己跑一遍"。文章末尾的"配套代码"链接指向这里对应的目录。代码里改进的地方（更清晰的命名、`--quick`、下载器）不会回写到文章的代码块里，两边数字若有出入以文章发表时的运行为准，并在该系列 README 里注明。

## License

MIT，见 [LICENSE](LICENSE)。文章本身是 CC BY 4.0，以博客页脚为准。
