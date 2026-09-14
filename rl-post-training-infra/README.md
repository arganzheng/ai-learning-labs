# RL 后训练基础设施：rollout 与训练如何共享一组 GPU — 配套代码

博客系列：[《RL 后训练基础设施：rollout 与训练如何共享一组 GPU》](https://arganzheng.life/rl-post-training-infrastructure.html)（八篇，Infra 地图 09）。贯穿脚本 `rl_ledger.py` 是"RL 一步账本"：第一篇建立三段（生成 / 打分 / 训练）的 FLOPs、显存与时间账，后面各篇给它加系统形态、切换、同步、异步等维度。

| 文件 | 文章 | 内容 | 依赖 |
|---|---|---|---|
| `rl_ledger.py` | [01 负载画像](https://arganzheng.life/rl-step-anatomy-rollout-reward-train.html) | 一步 GRPO / PPO 的 FLOPs 拆分、四份显存、decode 的带宽模型与长尾、全步 MFU；内置对话 / 推理 / MoE 三个场景 | 无 |

`expected/` 是各脚本的完整输出。所有数字都是理论估算（标称峰值、给定的 MFU 与带宽利用率假设），用于数量级判断与相互比较，不是实测。
