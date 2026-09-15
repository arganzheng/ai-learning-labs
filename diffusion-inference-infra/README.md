# 扩散模型推理基础设施：从一次去噪到一个生成服务 — 配套代码

博客系列：[《扩散模型推理基础设施：从一次去噪到一个生成服务》](https://arganzheng.life/diffusion-model-inference-infrastructure.html)（九篇，Infra 地图选修 12）。贯穿脚本 `diffusion_ledger.py` 是"一次生成的账本"：第一篇建立三段（文本编码器 / DiT × 步数 × CFG / VAE 解码）的 FLOPs、显存与时间账，后面各篇（单卡优化、跨步缓存、稀疏 attention、多卡并行、少步蒸馏、serving）都在这张账上做交换，用公式与表格记账，不再单独给脚本。

| 文件 | 文章 | 内容 | 依赖 |
|---|---|---|---|
| `diffusion_ledger.py` | [01 负载画像](https://arganzheng.life/diffusion-inference-workload-anatomy-and-cost-ledger.html) | token 数、每步线性 / attention FLOPs 与占比、三段的 FLOPs / 显存 / 时间、roofline 算术强度、与 7B LLM 生成 1000 token 的对照；内置 FLUX.1-dev / SD3-medium / Qwen-Image / Wan2.1-14B / HunyuanVideo，分辨率 / 帧数 / 步数 / CFG 扫描 | 无 |

```bash
python diffusion_ledger.py                                   # 五个模型的默认账 + 放大器扫描
python diffusion_ledger.py --model flux --gpu 4090           # 24 GB 卡上的 12B 模型
python diffusion_ledger.py --model wan --frames 129 --sweep  # 视频：帧数怎样放大 attention
python diffusion_ledger.py --model flux --steps 4 --no-cfg   # 少步蒸馏后的账
```

`expected/` 是默认运行的完整输出。所有数字都是理论估算（标称峰值 × 给定 MFU、经验的 VAE FLOP 密度），用于数量级判断与相互比较，不是实测；文章里凡引用实测（xDiT、SGLang Diffusion 的 benchmark）都单独注明来源。
