# 多模态：从视觉编码器到扩散模型 — 配套 toy 实验

博客系列：[总纲](https://arganzheng.life/multimodal-from-vision-encoders-to-diffusion.html)（算法地图 L7）。九篇正文各配一个脚本，**文中的核心代码、数字与图都由这些脚本跑出来**（图输出到 `out/*.svg`，博客里的 `img/in-post/multimodal-*.svg` 就是它们）。全部 CPU、离线、不下载任何模型；最慢的（06 / 07 / 08，训小 MLP 扩散模型）一两分钟。

| 脚本 | 文章 | 子实验 | 运行时间 |
|---|---|---|---|
| `01_vision_encoders_and_contrastive.py` | [视觉编码器](https://arganzheng.life/vision-encoders-clip-siglip-and-self-supervised-vit.html) | patchify infonce toy temperature sigmoid | 几秒 |
| `02_connectors_and_resolution.py` | [VLM 的结构](https://arganzheng.life/vlm-architecture-connectors-injection-and-dynamic-resolution.html) | connectors budget | 一秒 |
| `03_vlm_training_toys.py` | [VLM 的训练](https://arganzheng.life/vlm-training-recipe-data-stages-and-evaluation.html) | freeze cooccur | 十几秒 |
| `04_audio_mel_and_rvq.py` | [语音（上）](https://arganzheng.life/speech-and-omni-models-audio-encoders-codecs-and-duplex.html) | mel rvq | 几秒 |
| `05_duplex_timeline.py` | [语音（下）](https://arganzheng.life/speech-understanding-generation-and-full-duplex.html) | （示意图） | 一秒 |
| `06_ddpm_toy.py` | [扩散模型（上）](https://arganzheng.life/diffusion-models-ddpm-score-matching-and-flow-matching.html) | forward closedform train sample ddim | 一分钟 |
| `07_flow_score_cfg_toy.py` | [扩散模型（下）](https://arganzheng.life/score-matching-flow-matching-and-classifier-free-guidance.html) | score flow steps cfg（先跑 06 的 train） | 一两分钟 |
| `08_latent_diffusion_toy.py` | [Latent diffusion 与 DiT](https://arganzheng.life/latent-diffusion-dit-and-text-to-image-recipes.html) | vae latent cost | 一分钟 |
| `09_vq_tokenizer_and_ar_toy.py` | [自回归图像生成](https://arganzheng.life/autoregressive-image-generation-and-unified-models.html) | vq fsq ar steps | 几秒 |

公共文件：`_plot.py`（中文字体、博客列宽、SVG 输出——散点与箭头按 150 dpi 栅格化嵌入，几千个点的图也在 100 KB 左右）、`_diffusion_toy.py`（06 / 07 / 08 共用的两个月牙数据、小 MLP、DDPM 调度、训练循环）。

## 运行

```bash
pip install -r ../requirements.txt              # numpy / scikit-learn / matplotlib / torch（CPU 版即可）
python 06_ddpm_toy.py                            # 全部子实验；模型存到 out/06_ddpm_model.pt
python 07_flow_score_cfg_toy.py cfg              # 只跑一个
PLOT_PNG=/tmp/plots python 06_ddpm_toy.py        # 另存一份 png 方便自己看
```

数据全部是 scikit-learn 自带的（8×8 手写数字、`make_moons`）或合成的（随机特征、合成波形）。`expected/` 是作者机器上的完整输出；随机种子固定，不同版本下小数末位与耗时会有出入，结论应一致。
