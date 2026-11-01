# 大规模训练工程：从并行策略到容错恢复 — 配套实验

博客系列：[总纲](https://arganzheng.life/large-scale-training-from-parallelism-to-fault-tolerance.html)（Infra 地图 07）。

| 脚本 | 文章 | 子实验 | 运行时间 |
|---|---|---|---|
| `01_state_toys.py` | [训练任务的状态解剖](https://arganzheng.life/training-state-anatomy-memory-and-mfu.html) | bytes activations flops timeline | 秒级 |
| `01_state_figures.py` | 同上配图（out/training-state-*.svg → 博客 img/in-post/） | — | 秒级 |
| `02_parallelism_toys.py` | [并行策略全景](https://arganzheng.life/parallelism-strategies-which-state-to-shard.html) | zero tp cp/ring ulysses pp bubble ep mesh | 十秒 |
| `02_parallelism_figures.py` | 同上配图（out/parallelism-*.svg → 博客 img/in-post/） | — | 秒级 |
| `03_megatron_toys.py` | [Megatron-LM](https://arganzheng.life/megatron-lm-architecture-and-source-guide.html) | groups buffer 1f1b | 十秒 |
| `03_megatron_figures.py` | 同上配图（out/megatron-f*.svg → 博客 img/in-post/） | — | 秒级 |
| `04_deepspeed_toys.py` | [DeepSpeed](https://arganzheng.life/deepspeed-architecture-and-source-guide.html) | shards lifecycle memory | 十秒 |
| `04_deepspeed_figures.py` | 同上配图（out/deepspeed-f*.svg → 博客 img/in-post/） | — | 秒级 |
| `05_torchtitan_toys.py` | [torchtitan](https://arganzheng.life/torchtitan-architecture-and-source-guide.html) | mesh dtensor meta fsdp | 半分钟 |
| `05_torchtitan_figures.py` | 同上配图（out/torchtitan-f*.svg → 博客 img/in-post/） | — | 秒级 |
| `06_compare_toys.py` | [三框架对比](https://arganzheng.life/megatron-deepspeed-torchtitan-architecture-and-source-guide.html) | shard pp saw | 十秒 |
| `06_compare_figures.py` | 同上配图（out/compare-f*.svg → 博客 img/in-post/） | — | 秒级 |
| `07_config_toys.py` | [千卡配置与 MFU 调优](https://arganzheng.life/thousand-gpu-configuration-and-mfu-tuning.html) | config accum recompute | 十秒 |
| `07_config_figures.py` | 同上配图（out/config-*.svg → 博客 img/in-post/） | — | 秒级 |
| `08_ckpt_toys.py` | [分布式 checkpoint：格式、异步保存与重分片](https://arganzheng.life/distributed-checkpoint-format-async-save-and-resharding.html) | save reshard async young | 约三分钟（多进程 gloo） |
| `08_ckpt_figures.py` | 同上配图（out/distributed-checkpoint-{6tb-breakdown,young-formula,reshard-4to3}.svg → 博客 img/in-post/） | — | 秒级 |
| `09_ft_toys.py` | [容错与弹性](https://arganzheng.life/fault-tolerance-and-elastic-training.html) | mtbf watchdog elastic straggler | 约一分钟（多进程 gloo） |
| `09_ft_figures.py` | 同上配图（out/fault-tolerance-{five-terms,restart-lanes,replica-groups}.svg → 博客 img/in-post/） | — | 秒级 |
| `10_stab_toys.py` | [训练稳定性与数据管线](https://arganzheng.life/training-stability-and-data-pipeline.html) | spike clip logit bf16 packing resume | 约一分钟 |
| `10_stab_figures.py` | 同上配图（out/training-stability-{spike-rollback,clip-2d,logit-qkln,packing-vs-padding,resume-number-line}.svg → 博客 img/in-post/；读 expected/*.json） | — | 秒级 |
| `11_obs_toys.py` | [长时间训练的可观测性与运维](https://arganzheng.life/long-running-training-observability-and-operations.html) | fr timers ranks alert | 约半分钟（多进程 gloo） |
| `11_obs_figures.py` | 同上配图（out/long-running-training-{fr-rings,step-time-phases,rank-step-heatmap,alert-timeline}.svg → 博客 img/in-post/；读 expected/*.json） | — | 秒级 |

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

`expected/` 下是作者机器上的完整输出；随机种子固定，不同 torch 版本下小数末位可能有出入。
