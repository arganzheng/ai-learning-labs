#!/usr/bin/env bash
# 一次预训练从头走一遍：原料 → 过滤 → 去重 → tokenizer → 打包 → 选尺寸 → 训练 → 评。
# 数据（两个 Common Crawl WET 文件，约 215 MB）第一次会下载；MPS 上全程约两小时（step6 的 iso-FLOP 扫描占大半，加 --quick 缩到二十分钟），CPU 更久。
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
for s in 1_raw 2_filter 3_dedup 4_tokenizer 5_pack 6_isoflop 7_train 8_eval; do
  echo; echo "################ step$s ################"
  $PY step$s.py "$@" 2>&1 | grep -v -e findfont -e "num decayed" -e "num non-decayed" -e "using fused" -e "Loading weights" | tee expected/step$s.txt
done
