# coding-interview — 《面试手撕代码》配套代码

博客系列 [《面试手撕代码：从 LeetCode 中等题到 Transformer 组件》](https://arganzheng.life/coding-interview.html)（独立系列，不属于三张地图）的全部代码。文章里每一段代码都从这里复制，文中的输出、计数、对拍结果都由这里的脚本跑出来。

```
python/    01–13 算法题，Python，每篇一个文件，函数 + unittest（只用标准库）
java/      01–13 算法题，Java 21，每篇一个类，main 里 assert；make test
ai/        14–18 AI 岗手撕：NumPy 实现 + --check 用 torch 对拍
infra/     19 Infra 岗手撕：Python 并发题 + C++（make）
expected/  作者机器上的完整输出
```

## 运行

```bash
# Python（3.10+，标准库）
cd python && for f in p*.py; do python "$f"; done
python -m unittest discover -s python -p 'p*.py'

# Java（需要 JDK 21+；Makefile 默认找 /opt/homebrew/opt/openjdk，否则用 PATH 里的 javac）
cd java && make test
# Java 代码风格：google-java-format --aosp（4 空格缩进、一行一句、成员间空行）
java -jar google-java-format-all-deps.jar --aosp --replace java/P*.java

# AI 篇（numpy + torch，CPU）
cd ai && python attention.py --check

# Infra 篇
cd infra && make run
```

## 文章 ↔ 文件

| 篇 | 文章 | Python | Java |
|---|---|---|---|
| 01 | 数组、哈希与前缀和 | `p01_arrays_hashing.py` | `P01ArraysHashing.java` |
| 02 | 双指针与滑动窗口 | `p02_two_pointers_sliding_window.py` | `P02TwoPointersSlidingWindow.java` |
| 03 | 栈、单调栈与单调队列 | `p03_stack_monotonic.py` | `P03StackMonotonic.java` |
| 04 | 链表 | `p04_linked_list.py` | `P04LinkedList.java` |
| 05 | 二叉树 | `p05_binary_tree.py` | `P05BinaryTree.java` |
| 06 | 图：BFS / DFS / 拓扑 / 并查集 / 最短路 | `p06_graph.py` | `P06Graph.java` |
| 07 | 二分 | `p07_binary_search.py` | `P07BinarySearch.java` |
| 08 | 堆、Top-K、区间与贪心 | `p08_heap_intervals_greedy.py` | `P08HeapIntervalsGreedy.java` |
| 09 | 回溯 | `p09_backtracking.py` | `P09Backtracking.java` |
| 10 | 字符串 | `p10_strings.py` | `P10Strings.java` |
| 11 | 动态规划（一）：线性与二维 | `p11_dp_linear_grid.py` | `P11DpLinearGrid.java` |
| 12 | 动态规划（二）：背包、区间、状态机、树形 | `p12_dp_knapsack_interval_state.py` | `P12DpKnapsackIntervalState.java` |
| 13 | 设计题与数据结构实现 | `p13_design.py` | `P13Design.java` |

| 篇 | 文章 | 文件 | 运行 |
|---|---|---|---|
| 14 | 手撕 attention 家族 | `ai/attention.py` | `python attention.py [--check]` |
| 15 | 手撕 Transformer block 与反向传播 | `ai/transformer_block.py` | 同上 |
| 16 | 手撕 tokenizer 与解码 | `ai/tokenizer_decoding.py` | 同上 |
| 17 | 手撕损失函数与训练算法 | `ai/losses_training.py` | 同上 |
| 18 | 手撕经典 ML 与评测指标 | `ai/classical_ml_metrics.py` | 同上（NMS 对拍需要 torchvision，可选） |
| 19 | Infra 岗手撕：并发与系统 | `infra/concurrency.py`、`infra/memory_pool.cpp`、`infra/blocked_gemm.cpp`、`infra/topk_lru.cpp` | `make run` |

`ai/*.py` 不带参数打印文章里引用的数值示例（形状推演、参数量、采样分布…），`--check` 用 float64 与 torch 的参考实现对拍（`F.scaled_dot_product_attention`、`nn.MultiheadAttention`、`F.layer_norm` 的 autograd、`F.cross_entropy(label_smoothing, ignore_index)`、`torch.optim.AdamW`、`clip_grad_norm_`、`F.conv2d` 等），容差 1e-10。`infra/blocked_gemm` 的计时随机器变化，`expected/infra.txt` 是 Apple M 系列 `-O2` 的结果——在这台机器上简单分块**慢于** ikj，文章据实报告并解释了原因。

## 选题打分表

每篇的主讲题按三条打分（各 0–2 分）：**高频**（LeetCode Hot 100 / 剑指 Offer / CodeTop 近一年大厂频次表可查）、**模板代表性**（一道题逼出该模式的全部要点，而不是模式的特例）、**follow-up 空间**（面试官能顺着追问一到两层）。5–6 分主讲，3–4 分进题单给一句提示，更低的不收。

| 篇 | 主讲（分） | 题单（分） | 落选与原因 |
|---|---|---|---|
| 01 | 560 前缀和+哈希 (6) · 128 (5) · 41 原地哈希 (5) · 238 (5) · 1109 差分 (4→主讲，差分唯一代表) | 1 · 136 · 260 · 287 | 303 区间和检索（太直接）、442（与 41 同技巧） |
| 02 | 3 (6) · 76 (6) · 15 (6) · 42 (6) · 424 (5) · 11 (5) | 209 · 567 · 283 · 167 | 18 四数之和（15 的机械推广） |
| 03 | 20 (5) · 394 (5) · 739 (5) · 84 (6) · 239 (6) · 227 (5) | 150 · 155 · 85 · 42 单调栈版 | 224 含括号计算器（227 加递归即可，正文提一句） |
| 04 | 206 (6) · 92 (5) · 25 (6) · 142 (6) · 23 (5) · 148 (5) | 21 · 19 · 160 · 876 · 234 · 138 | 2 两数相加（进位模拟，无链表要点） |
| 05 | 102 (5) · 236 (6) · 105 (5) · 124 (6) · 98 (5) · 297 (5) · 437 (5) | 94/144 · 104 · 226 · 101 · 543 · 230 · 114 · 199 | 112/113 路径和（437 覆盖） |
| 06 | 200 (6) · 994 (5) · 207/210 (6) · 127 (5) · 721 (5) · 743 (5) | 695 · 133 · 547 · 1091 | 785 二分图（频次低）、787 有 k 站（Bellman-Ford，正文提要） |
| 07 | 34 (5) · 33 (6) · 153 (5) · 875 (5) · 410 (6) · 4 (6) · 378 (5) | 35 · 81 · 162 · 69 · 1011 | 74/240 搜索二维矩阵（378 阶梯法覆盖） |
| 08 | 215 (6) · 347 (5) · 295 (6) · 56 (6) · 253 (6) · 435 (5) · 45 (5) | 973 · 57 · 452 · 621 · 55 · 134 · 122 | 692 前 K 个高频单词（347 加比较器） |
| 09 | 46/47 (6) · 78/90 (6) · 39/40 (6) · 22 (5) · 131 (5) · 79 (5) · 51 (5) | 17 · 77 · 93 | 37 数独（51 的推广，代码长而无新要点） |
| 10 | 5 (6) · 28 KMP (5) · 8 (5) · 43 (5) · 179 (5) · 187 (4→主讲，滚动哈希唯一代表) | 647 · 49 · 415 · 151 · 14 · 443 | 76（已在 02）、14（太简单） |
| 11 | 322 (6) · 300 (6) · 53/152 (5) · 1143 (5) · 72 (6) · 221 (5) · 139 (5) | 70 · 198/213 · 62 · 64 · 91 | 279 完全平方数（322 同型） |
| 12 | 416 (6) · 518 (5) · 312 (6) · 188 (6) · 309 (5) · 337 (5) · 10/44 (5) | 494 · 516 · 121/122/123 · 714 · 847 | 1049 最后一块石头（416 变体） |
| 13 | 146 (6) · 460 (6) · 208/212 (5) · 380 (5) · 307 (4→主讲，树状数组唯一代表) | 155 · 232 · 355 · 981 | 1206 跳表（频次低，正文提要） |

## 与文章的关系

代码是附件：文章负责讲清模式、推演与陷阱，代码只负责"你可以自己跑、改着玩"。Python 与 Java 的写法尽量一一对应，差异之处（`Deque` 而不是 `Stack`、`Integer.compare` 而不是相减、`long` 防溢出、`lo + (hi - lo) / 2`）在正文的「两种语言的坑」一节列出。
