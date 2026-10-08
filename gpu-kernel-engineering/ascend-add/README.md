# Ascend C Add：从索引账本到官方 Kernel Launch 工程

对应文章：[GPU Kernel 工程（11）](https://arganzheng.life/ascend-cann-from-cuda-to-ascend-c.html)。

## 版本与实现边界

- 文档基线：CANN **8.0.0 商业版** [Kernel Launch 指南](https://www.hiascend.com/document/detail/en/canncommercial/800/opdevg/Ascendcopdevg/atlas_ascendc_10_0005.html)。该页链接到下列官方样例 tag，而非 GitHub 上停留在 2023 年的镜像。
- 样例：[Ascend/samples `v0.2-8.0.0.beta1`](https://gitee.com/ascend/samples/tree/v0.2-8.0.0.beta1/operator/ascendc/0_introduction/3_add_kernellaunch/AddKernelInvocationNeo)，commit `46e58792cd0732702645e0b1c274165c4f1abc65`（2024-12-20）。`vendor/` 原样收录此目录全部 10 个文件，另附上游 LICENSE / NOTICE；版权归 Huawei，Apache-2.0，不随本仓 MIT 许可证改变。未自行重写上游 CMake / runtime API。
- 讲解和命令选择的 SoC：**Ascend910B1**（Atlas A2）；必须用 `npu-smi info` 确认自己的型号，不要把 `Ascend910B` 当成 `Ascend910B1` 的缩写。
- 固定输入：两个 contiguous ND `float16[8, 2048]`，输出同形状；8 个 block，每块 2048 元素，每次 128 元素，共 16 轮；无广播、stride、任意长度或尾块支持。
- 本次验证环境：Linux x86_64 / Python 3.10 / NumPy 2.2.6，**没有 CANN 和 NPU**。仅跑过下述 NumPy 账本及验证器；没有编译或运行 Ascend C，不提供真实性能数据。硬件、驱动、固件和 CANN 必须按官方配套表安装，不能据本目录声称其它组合已验证。

## 1. 无 CANN：核对分块账本

在本目录运行：

```bash
python3 reference.py audit
```

它检查负数、零、每个 tile / block 的首尾哨兵、覆盖次数和不支持输入的拒绝路径，并算出 1536 B 队列 payload、98304 B 逻辑全局读写量。**NumPy 的切片相加不是 Ascend C CPU 调试，也不能验证队列同步。**

## 2. 有 CANN：原样编译并运行官方工程

安装 CANN 8.0.0 开发包和相配的驱动/固件（以及 CPU 调试所需组件），准备 CMake ≥3.16、C++ 编译器及 Python/NumPy。先配置自己的实际安装目录：

```bash
export ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/8.0.0
source "$ASCEND_INSTALL_PATH/bin/setenv.bash"
cd vendor
bash run.sh -r cpu -v Ascend910B1
# 有 Ascend910B1 真机时，改用：
bash run.sh -r npu -v Ascend910B1
```

路径是示例，`ASCEND_INSTALL_PATH` 必须指向实际版本，不能让 `latest` 静默切版本。脚本会重新生成 `build/ out/ input/ output/`，在 **vendor 目录**执行。CPU 模式通过 `ICPU_RUN_KF`，NPU 模式经 `add_custom_do` 的 `<<<blockDim, nullptr, stream>>>` 启动，再 `aclrtSynchronizeStream`。官方 README 中提到的 `ACLRT_LAUNCH_KERNEL` 并不是该 revision 的实际代码路径，应以 `main.cpp` / `add_custom.cpp` 为准。

默认脚本生成随机正数并用上游验证器验收；其容许一定比例元素不匹配。文章的严格验收另用我们的验证器，要求**每个元素**满足容差，且长度和有限性都正确：

```bash
python3 ../reference.py verify --root .
```

## 3. 确定性边界输入

先用上一步完成相应模式的编译；保持相同 CANN 环境和动态库搜索路径（CPU 还需要 `tikicpulib` / simulator 库）。然后在 vendor 目录执行，**不要再运行 run.sh，它会覆盖输入**：

```bash
export LD_LIBRARY_PATH="$PWD/out/lib:$PWD/out/lib64:$ASCEND_INSTALL_PATH/lib64:$ASCEND_INSTALL_PATH/tools/tikicpulib/lib:$ASCEND_INSTALL_PATH/tools/tikicpulib/lib/Ascend910B1:$ASCEND_INSTALL_PATH/tools/simulator/Ascend910B1/lib:${LD_LIBRARY_PATH:-}"
for case in signed zeros boundaries; do
    python3 ../reference.py generate --case "$case" --root .
    ./ascendc_kernels_bbit
    python3 ../reference.py verify --root . || exit 1
done
```

这只在固定 shape 内改变值和检查切分边界。`N=0/1/127/129/16385`、非连续输入、dtype 改变等是后续**泛化实现**必须覆盖的输入，不支持直接喂给本样例，也不靠补一个 if 就自动解决 DMA 对齐。

## 4. 性能验证（未执行）

官方脚本已有 `RUN_WITH_TOOLCHAIN=1` 分支：

```bash
RUN_WITH_TOOLCHAIN=1 bash run.sh -r npu -v Ascend910B1
```

它使用 CANN 的 `msprof op --application=./ascendc_kernels_bbit`。单次调用只够观察时间线，不能当稳定吞吐 benchmark。正式测试另加 warmup / 重复、设备侧计时与统计，报告 SoC、CANN、输入尺寸、编译选项、缓存状态；分开 H2D/D2H、kernel 和端到端时间。固定 32 KiB 输入很小，可能受 launch / cache 影响，不适合声称达到 HBM 峰值。大输入必须先改 tiling 与 host 分配并通过正确性验证。

## 本目录维护检查

新增 Python：`ruff check reference.py`、`mypy --strict reference.py`、`python3 reference.py audit`；上游工程只做 `bash -n vendor/run.sh`、逐文件原样比对，不把缺少 CANN 的状态标成编译通过。
