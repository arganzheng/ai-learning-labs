# C++ 在 AI-Infra：从对象模型到算子扩展 — 配套代码

博客系列：[总纲](https://arganzheng.life/cpp-for-ai-infra.html)。

```bash
make          # 编译全部（默认 c++ -std=c++17 -Wall -Wextra；CXX=g++ make 换编译器）
make run      # 依次运行行为正常的例子
make ex04     # 单独编译"故意编不过"的 ex04，看 const 正确性的报错
make clean
```

## 02 [值、引用与所有权——对象模型与 RAII](https://arganzheng.life/cpp-value-semantics-ownership-and-raii.html)

`02-value-semantics-and-raii/`，每个文件对应文中一个小节，几十行，`c++ -std=c++17 exNN.cpp && ./a.out` 即可单独运行。

| 文件 | 演示 |
|---|---|
| `ex01_stack.cpp` | 值语义：拷贝后互不影响 |
| `ex02_tracer_copy.cpp` | 用打印构造 / 析构的 `Tracer` 看拷贝发生在哪里 |
| `ex03_ref.cpp` | 引用与指针：引用不能改绑，`r = b` 是赋值 |
| `ex04_const_err.cpp` | **故意编不过**：忘记 `const` 成员函数、`const int*` vs `int* const` |
| `ex05_buffer_bad.cpp` | **故意 double free**：编译器生成的拷贝构造做浅拷贝 |
| `ex06_buffer_good.cpp` | Rule of five：深拷贝 + 移动 |
| `ex07_move.cpp` | 左值 / 右值的重载决议、移动后的对象状态、返回值优化、`return std::move(x)` 反而阻碍 RVO（编译器会警告） |
| `ex08_vector_noexcept.cpp` | `vector` 扩容时只有 `noexcept` 的移动构造才会被用 |
| `ex09_raii.cpp` | RAII 与异常：栈展开时析构一定执行 |
| `ex10_shared.cpp` | `unique_ptr` / `shared_ptr` / `weak_ptr` 的大小与引用计数 |
| `ex11_cycle.cpp` | `shared_ptr` 循环引用泄漏，`weak_ptr` 打破 |
| `ex12_unique_deleter.cpp` | 带自定义 deleter 的 `unique_ptr` 管 `malloc` 内存 |
| `ex13_dangling.cpp` | **未定义行为**：返回局部变量的引用 |
| `ex14_dtor_order.cpp` | 构造 / 析构顺序：基类、成员、派生类 |

## mini-c10：系列贯穿的迷你 Tensor 运行时

`minic10/` 是按 PyTorch c10 的结构缩写的一套头文件，本篇（02）实现所有权链：`Tensor → intrusive_ptr<TensorImpl> → intrusive_ptr<StorageImpl> → DataPtr(deleter) → Allocator`。

```text
minic10/util/intrusive_ptr.h   80 行的侵入式引用计数（文中第九章）
minic10/core/Allocator.h       Allocator / DataPtr / CPUAllocator
minic10/core/StorageImpl.h     一段带 deleter 的内存
minic10/core/TensorImpl.h      sizes / strides / dtype / storage
minic10/core/Tensor.h          值语义的句柄
minic10/core/ScalarType.h      dtype 枚举（第 3 篇补类型映射）
minic10/core/DispatchKey.h     占位（第 4 篇的内容）
minic10/main.cpp               走一遍拷贝 / 移动 / 作用域，打印 use_count 与 malloc / free
minic10/test_intrusive.cpp     intrusive_ptr 的基本行为
```

`make` 产出 `build/minic10_tensor` 与 `build/minic10_intrusive`。后续文章（模板、多态、宏与注册、并发、pybind11）会在这套骨架上继续加文件。
