# math-for-ai —《算法工程师的数学》配套代码

博客系列 [《算法工程师的数学：从矩阵形状到策略梯度》](https://arganzheng.life/math-for-ai-algorithm-engineers.html)（算法地图 L0）里"有例子、有图、有实跑数字"的那几篇，其全部数字与图由这里的脚本生成。数学系列的前四篇与第六篇以手算为主，只有第 01 篇第六章的 attention 投影玩具有脚本。

| 脚本 | 文章 | 内容 | 依赖 |
|---|---|---|---|
| `01_attention_projections.py` | [01 向量、矩阵与形状](https://arganzheng.life/vectors-matrices-shapes-and-flops.html) 第六章 | 二维玩具：不投影时 attention 分数对称、只给自己高分，$W_Q W_K^T$ 投影后"它"看"猫"；两个头的 $W_O$ 分块写回残差流（配图）；Llama-3-8B 合并 $W_V W_O$ 的参数账 | numpy |
| `07_gradients_and_policy_gradient.py` | [07 导数、梯度与链式法则](https://arganzheng.life/derivatives-gradients-chain-rule-and-policy-gradient.html) | 割线→切线；梯度箭头场；链式法则手算 / autograd / 有限差分三方对拍与计算图；softmax+CE 梯度 = p − y；**10 个 token 的玩具策略**：精确梯度 vs REINFORCE 估计的分布、baseline 与离谱 baseline、GRPO 的 (1 − 1/G) 偏差精确枚举、四种方法的训练曲线（含奖励抬到 10/11 时 REINFORCE 坍缩）；三种学习率的 GD 轨迹与 SGD 抖动；拉格朗日相切图 | numpy、torch、matplotlib |
| `07_rl_on_nanogpt.py` | 同上，第八章案例 | 在第四篇训好的莎士比亚 nanoGPT 上做最小的策略梯度：奖励 = 元音比例，GRPO 骨架，β = 0 时 10 步内 reward hacking 成 `iiii…`、val loss 1.72 → 6.8；β = 0.5 的 KL 惩罚下奖励 0.38 → 0.61、文本仍是英文 | torch（MPS 或 CPU）；需要 `../transformer-and-llm/nanogpt/out-shakespeare-char-base/ckpt.pt`（按 Transformer 04 训一次，约 7 分钟） |
| `08_inference_and_fitting.py` | [08 统计推断与拟合](https://arganzheng.life/statistical-inference-and-fitting-scaling-laws.html) | 模拟：同一模型评 1000 次的抽样噪声、CLT、100 个置信区间、独立 vs 配对比较、置换检验；5 点最小二乘手算；用 `transformer-and-llm/scaling_law_fit.py` 的 7 个真实模型拟幂律 + bootstrap 区间 + 外推；Chinchilla 等 loss 线 / 等算力线 | numpy、scipy、matplotlib |
| `08_multi_seed.py` | 同上，第四章 | 实跑 4 个配方 × 5 个种子（0.1M 参数小 GPT，400 步）：种子噪声多大、+2% / +20% 学习率与 weight decay 改动能否判定 | torch（CPU 约 5 分钟）；需要莎士比亚数据 |
| `05_mle_to_cross_entropy.py` | [05 从最大似然到交叉熵](https://arganzheng.life/from-maximum-likelihood-to-cross-entropy.html) | 硬币的似然曲线；三个 token 的玩具语言模型；下溢演示；−log p 曲线；softmax 手算 / 对拍 / 溢出；nanoGPT 首步 loss ≈ ln V 与三种 bug；GPT-2 上的温度 / top-k / top-p；Qwen2.5 chat 模板的 loss mask；奖励模型曲线；读一条训练日志 | numpy、torch、matplotlib；`sampling` / `mask` 两段需 transformers 与本地缓存的 `gpt2`、`Qwen/Qwen2.5-0.5B`（只用 tokenizer） |

## 运行

```bash
cd math-for-ai
python 01_attention_projections.py out/01-w-o-multi-head.svg   # 几毫秒
python 05_mle_to_cross_entropy.py              # 全部段，约 1–2 分钟（CPU）
python 05_mle_to_cross_entropy.py coin lm      # 只跑指定的段；段名见脚本 docstring
python 07_gradients_and_policy_gradient.py     # 约 1 分钟
python 07_rl_on_nanogpt.py                     # MPS 约 2 分钟 / CPU 约 6 分钟；--quick 各 20 步
python 08_inference_and_fitting.py             # 约 30 秒
python 08_multi_seed.py                        # 20 次训练，CPU 约 5 分钟
PLOT_PNG=/tmp/plots python 05_mle_to_cross_entropy.py   # 另存一份 png 便于自查
```

图输出到 `out/*.svg`（博客 `img/in-post/math-*.svg` 就是这些文件），文字输出与 `expected/*.txt` 对照。`05` 的 `lnv` 段、`07_rl_on_nanogpt.py`、`08_multi_seed.py` 读取 `../transformer-and-llm/nanogpt/data/shakespeare_char/` 的数据与 `../transformer-and-llm/expected/train_shakespeare_char_base.txt` 的日志，需要先按 `transformer-and-llm/README.md` 准备莎士比亚数据（`prepare.py`）。
