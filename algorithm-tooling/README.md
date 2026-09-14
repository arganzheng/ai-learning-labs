# 算法工程师的工具箱：从一个想法到一次能跑的实验 — 配套脚本

博客系列：[总纲](https://arganzheng.life/tooling-for-ai-algorithm-engineers.html)（算法地图 L1）。五篇文章各配一个脚本，文中引用的数字由这些脚本跑出来。全部在 CPU 上可跑。

| 脚本 | 文章 | 依赖 | 运行时间（8 核笔记本 CPU） |
|---|---|---|---|
| `01_numpy_pandas_matplotlib.py` | [科学计算栈](https://arganzheng.life/numpy-pandas-matplotlib-for-algorithm-engineers.html) | numpy、torch（对数值）、pandas、matplotlib | 几秒；图存到 `out/loss_curves.png` |
| `02_train_loop.py` | [PyTorch 使用层（上）](https://arganzheng.life/pytorch-in-use-five-objects-and-a-training-loop.html) | torch | 1000 步约 1 分钟；`--quick` 100 步 |
| `03_memory_ledger.py` | [PyTorch 使用层（下）](https://arganzheng.life/pytorch-in-use-mixed-precision-memory-ledger-and-multi-gpu.html) | torch | 1 秒；有 CUDA 时多一节实测对账 |
| `04_hf_lora_sft.py` | [Hugging Face 生态](https://arganzheng.life/hugging-face-ecosystem-six-libraries-and-a-lora-sft.html) | transformers、peft、trl、datasets；**首次运行下载 Qwen2.5-0.5B（约 1 GB）** | 20 步约 1 分钟（含加载）；`--quick` 5 步 |
| `05_profiler_and_record.py` | [GPU 直觉与实验管理](https://arganzheng.life/gpu-intuition-and-experiment-management.html) | torch | 十几秒；记录写到 `out/run-*.json` |

`tinygpt.py` 是 02 / 05 共用的字符级小 Transformer（4 层、d=128、84 万参数），语料是本机 Python 标准库的源码，不用下载。

## 运行

```bash
pip install -r ../requirements.txt        # numpy torch pandas matplotlib（04 还要 transformers peft trl datasets）
python 01_numpy_pandas_matplotlib.py      # 全部子实验
python 01_numpy_pandas_matplotlib.py attention pandas
python 02_train_loop.py --quick
python 05_profiler_and_record.py -h 2>/dev/null || python 05_profiler_and_record.py profile
```

`expected/` 是作者机器上（Apple Silicon，PyTorch CPU）的完整输出。不同 CPU / BLAS / 版本下小数末位、耗时会不同，趋势与结论应一致。

两个与文章有关的注：

- 02 / 05 的训练循环里 `torch.autocast(..., enabled=DEV == "cuda")`：bf16 在没有硬件支持的 CPU 上反而慢 30 倍（作者机器上 1.4 s/步 vs 0.05 s/步），所以只在 CUDA 上开；文章里的二十行循环按 GPU 写，默认开着。
- 04 用 12 条问答训 20 步只是为了看清 chat template、loss mask、LoRA 挂载与训练状态这几件事；生成结果学会了答案但还没学会在 `<|im_end|>` 停下，属于预期（数据太少、步数太少），文章里有说明。
