# 大规模训练工程：从并行策略到容错恢复 — 配套实验

博客系列：[总纲](https://arganzheng.life/large-scale-training-from-parallelism-to-fault-tolerance.html)（Infra 地图 07）。

| 脚本 | 文章 | 子实验 | 运行时间 |
|---|---|---|---|
| `02_parallelism_toys.py` | [并行策略全景](https://arganzheng.life/parallelism-strategies-which-state-to-shard.html) | zero tp cp pp ep | 十秒 |

**没有 GPU 也能跑。** `torch.distributed` 的 gloo 后端在 4 个 CPU 进程之间做 all-reduce / reduce-scatter / all-gather / all-to-all / send-recv，语义与 NCCL 在 GPU 之间完全相同，只是慢。每个子实验都把并行版的结果与「单进程算一遍」的参考值对数：

| 子实验 | 真跑的是什么 | 验证 |
|---|---|---|
| `zero` | 8 个梯度元素：all-reduce vs reduce-scatter + 更新 + all-gather | 两条路结果相同——ZeRO-1/2 通信量不变的那条等式 |
| `tp` | X(1×2) → A 列切 → ReLU → B 行切，N_t = 2；`f` / `g` 两个 autograd.Function 与 Megatron `mappings.py` 同构 | 前向 Z、反向 dA / dB / dX 与单卡逐格相等（数字小到能手算） |
| `cp` | 8 个 token 切 4 卡，K/V 沿环 `isend` / `recv` 3 步，online-softmax 合并 | 与完整注意力差 1e-7 |
| `pp` | 4 层各放一个进程，4 个 micro-batch，GPipe 顺序 | 各 stage 权重梯度与单卡差 1e-7 |
| `ep` | 8 个专家、k = 2、4 进程各持 2 个；token 按路由 all-to-all 去专家所在进程再回来；反向是 split 互换的 all-to-all | 前向误差 0、梯度差 1e-6；同时打印 dispatch 矩阵、每进程负载 ρ、容量因子 1.25 下溢出的专家 |

```bash
pip install -r ../requirements.txt          # 只需要 torch（CPU 版即可）
python 02_parallelism_toys.py               # 全部
python 02_parallelism_toys.py tp ep         # 只跑几个
```

`expected/02_parallelism_toys.txt` 是作者机器上的完整输出；随机种子固定，不同 torch 版本下小数末位可能有出入。
