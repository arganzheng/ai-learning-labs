# 深度学习基础：从反向传播到残差 — 配套实验

博客系列：[总纲](https://arganzheng.life/deep-learning-foundations.html)。六篇文章各配一个脚本，文中的每个数字都由这些脚本跑出来。

| 脚本 | 文章 | 依赖 | 完整运行 |
|---|---|---|---|
| `01_backprop.py` | [手推反向传播](https://arganzheng.life/backpropagation-by-hand.html) | NumPy（`torch` 子实验可选） | 5 s |
| `02_init_norm_residual.py` | [初始化、归一化与残差](https://arganzheng.life/initialization-normalization-and-residual.html) | NumPy | 1 min |
| `03_optimizers.py` | [优化器：从 SGD 到 AdamW](https://arganzheng.life/optimizers-from-sgd-to-adamw.html) | NumPy | 3 min |
| `04_regularization.py` | [正则化与泛化](https://arganzheng.life/regularization-and-generalization.html) | NumPy | 15 min |
| `05_cnn.py` | [CNN：从 LeNet 到 ResNet 与 ViT](https://arganzheng.life/cnn-from-lenet-to-resnet-and-vit.html) | PyTorch（`resnet` 子实验需 torchvision） | 10 min |
| `06_rnn_attention.py` | [RNN、LSTM 与 attention 的诞生](https://arganzheng.life/rnn-lstm-and-the-birth-of-attention.html) | PyTorch | 6 min |

时间是 8 核笔记本 CPU 上的量级，不需要 GPU。

## 案例与图

| 脚本 | 文章 | 产出 |
|---|---|---|
| `case_01_mnist_mlp.py` | 01 | MNIST 两层 MLP 15 个 epoch：训练曲线、错分样本（`out/case-01-*.svg`） |
| `case_02_deep_mlp.py` | 02 | 64 层 MLP 七种接法：逐层激活 std / 梯度范数曲线、300 步 loss 曲线 |
| `case_03_optimizers_mnist.py` | 03 | 四种优化器 × 五个学习率扫描、warmup 在 64 层网络上的曲线、调度形状 |
| `case_04_regularization_plots.py` | 04 | 1,000 张图四种正则化的 train / test 曲线；宽度扫描的 double descent（约 10 分钟） |
| `case_05_lenet5.py` | 05 | 复现 LeNet-5：逐层形状与参数、5 个 epoch 到 99.18%、第一层卷积核与特征图、错分样本（PyTorch，MPS / CUDA / CPU） |
| `case_06_char_lstm_and_alignment.py` | 06 | 字符级 LSTM 写莎士比亚（与 ../transformer-and-llm/nanogpt 同一语料同预算，val 1.71 vs 1.66）；seq2seq + attention 的对齐矩阵热力图 |
| `tools/paper_figures.py` | 02–06 | 从原论文 PDF 裁出经典结构图（Pre-LN vs Post-LN、Adam Algorithm 1 …）到 `out/paper-*.png`，博客侧转 WebP 并注明版权 |

`_plot.py` 是画图的公共设置（与 `classical-ml/_plot.py` 相同）。

## 运行

```bash
pip install -r ../requirements.txt        # numpy 即可跑 01–04；05/06 需要 torch
python 01_backprop.py                     # 全部子实验，数字对照文章
python 01_backprop.py --quick             # 一分钟内跑完，用来确认环境
python 03_optimizers.py adamw scaling     # 只跑指定的子实验
python 03_optimizers.py -h                # 列出子实验
```

第一次运行会把 MNIST（约 12 MB）下载到 `data/`。`expected/` 目录是作者机器上完整运行的输出，不同 CPU / BLAS / PyTorch 版本下小数末位与耗时会不同，趋势和结论应该一致。

## 结构

```text
dlf/            系列贯穿的迷你框架（NumPy）
  nn.py         Linear / ReLU / softmax-CE / MLP、梯度检查、FLOPs 计数        ← 01
  layers.py     LayerNorm / RMSNorm / Residual / Dropout / Embedding、深层 MLP  ← 02, 04
  optim.py      SGD / Momentum / Adam / AdamW、warmup+cosine、梯度裁剪          ← 03
  data.py       MNIST 下载与加载、字符级语料
  cli.py        --quick 与子实验选择
0X_*.py         每篇文章的实验脚本，只依赖 dlf 与（05/06）torch
expected/       作者机器上的完整输出
```

## 动手扩展

每篇文章末尾都有"值得自己动手的扩展"，都可以在这些脚本上直接改：换激活函数看梯度检查是否仍通过（01）、把 `prenorm` 换成 RMSNorm（02）、给 SGD 加 cosine schedule 与 Adam 对比（03）、扫训练步数看 epoch-wise double descent（04）、把 `ConvNet` 的通道数加倍看 plain/residual 差距（05）、在 seq2seq 上换成 dot-product attention（06）。
