# math-for-ai —《算法工程师的数学》配套代码

博客系列 [《算法工程师的数学：从矩阵形状到策略梯度》](https://arganzheng.life/math-for-ai-algorithm-engineers.html)（算法地图 L0）里"有例子、有图、有实跑数字"的那几篇，其全部数字与图由这里的脚本生成。数学系列的前四篇与第六篇以手算为主，暂无脚本。

| 脚本 | 文章 | 内容 | 依赖 |
|---|---|---|---|
| `05_mle_to_cross_entropy.py` | [05 从最大似然到交叉熵](https://arganzheng.life/from-maximum-likelihood-to-cross-entropy.html) | 硬币的似然曲线；三个 token 的玩具语言模型；下溢演示；−log p 曲线；softmax 手算 / 对拍 / 溢出；nanoGPT 首步 loss ≈ ln V 与三种 bug；GPT-2 上的温度 / top-k / top-p；Qwen2.5 chat 模板的 loss mask；奖励模型曲线；读一条训练日志 | numpy、torch、matplotlib；`sampling` / `mask` 两段需 transformers 与本地缓存的 `gpt2`、`Qwen/Qwen2.5-0.5B`（只用 tokenizer） |

## 运行

```bash
cd math-for-ai
python 05_mle_to_cross_entropy.py              # 全部段，约 1–2 分钟（CPU）
python 05_mle_to_cross_entropy.py coin lm      # 只跑指定的段；段名见脚本 docstring
PLOT_PNG=/tmp/plots python 05_mle_to_cross_entropy.py   # 另存一份 png 便于自查
```

图输出到 `out/*.svg`（博客 `img/in-post/math-*.svg` 就是这些文件），文字输出与 `expected/*.txt` 对照。`05` 的 `lnv` 段读取 `../transformer-and-llm/nanogpt/data/shakespeare_char/` 的数据与 `../transformer-and-llm/expected/train_shakespeare_char_base.txt` 的日志，需要先按 `transformer-and-llm/README.md` 准备莎士比亚数据（`prepare.py`）。
