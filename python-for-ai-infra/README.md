# Python 在 AI-Infra：从语言机制到生产交付 — 配套代码

博客系列：[总纲](https://arganzheng.life/python-for-ai-infra.html)。这些脚本是写文章时用来验证每个结论的最小实验，只依赖标准库，`python3 xxx.py` 直接跑；输出就是文章里引用的行为。

## 01 [语言机制与运行时内幕](https://arganzheng.life/python-language-mechanisms-and-runtime-internals.html)

| 脚本 | 内容 |
|---|---|
| `01-language-mechanisms/code_objects_and_scopes.py` | `dis`、code object、函数对象、frame 与 traceback、LEGB、闭包晚绑定、`nonlocal`、`UnboundLocalError`、小整数缓存 |
| `01-language-mechanisms/import_system.py` | `__spec__` / loader、内建与扩展模块、`sys.meta_path`、`path_hooks`、`find_spec`、`sys.modules` 身份 |
| `01-language-mechanisms/meta_path_finder.py` | 自定义 `MetaPathFinder`：屏蔽一个模块、追踪 import 请求 |
| `01-language-mechanisms/object_model_and_descriptors.py` | `type` 与元类、数据 / 非数据描述符优先级、bound method、`__getattr__` vs `__getattribute__`、`__new__`、MRO、`__eq__` 使 `__hash__` 失效、`__slots__` |
| `01-language-mechanisms/generators_context_managers_exceptions.py` | property 里的 `AttributeError` 被 `__getattr__` 吞掉、`super()` 的 `__class__` cell、生成器状态与 `close()`、`with` 的展开、`contextmanager` 异常注入、异常链、`finally` |
| `01-language-mechanisms/decorators_and_callables.py` | 装饰 `classmethod` 的 `TypeError`、类装饰器丢失 `self`、用 `__get__` 修好 |

## 03 [并发、异步与任务协作](https://arganzheng.life/python-concurrency-asynchrony-and-task-collaboration.html)

| 脚本 | 内容 |
|---|---|
| `03-concurrency/thread_pool_semantics.py` | `Future.cancel` 只对未开始的任务有效、`result(timeout)` 不停 worker、`shutdown(cancel_futures=True)`、`Event` 停止标志、`threading.local` vs `contextvars` 在 `asyncio.to_thread` 中的行为、有界 `queue.Queue` |
| `03-concurrency/process_pool_semantics.py` | 进程池的 cancel 语义、`contextvars` 不跨进程、worker 被杀 → `BrokenProcessPool`、有界 `mp.Queue` |

字节码与 `dis` 输出随 CPython 版本变化（这里用 3.12 跑），其他行为在 3.10+ 上一致。
