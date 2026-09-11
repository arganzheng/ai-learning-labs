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
