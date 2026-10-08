# GPU Kernel 工程

第 11 篇：[从 CUDA 到昇腾——用 Ascend C 重写一个算子](https://arganzheng.life/ascend-cann-from-cuda-to-ascend-c.html)。

- [`ascend-add/`](ascend-add/)：锁定官方版本的完整 Add 工程、确定性输入、逐元素验证器和无需 CANN 的分块账本。
- CUDA 主线的其它实验尚未收录；本目录不是完整系列的 GPU benchmark。
- 普通 CPU 只能运行 NumPy 账本；Ascend C CPU 调试仍需 CANN 的 `tikicpulib`，NPU 执行需匹配的昇腾设备及驱动。
