"""面试手撕代码（19）：Infra 岗手撕 —— 并发与系统题（Python 部分）。

https://arganzheng.life/coding-interview-infra-concurrency-and-systems.html

    python concurrency.py    # 跑全部自检并打印关键数字
"""
from __future__ import annotations

import queue
import threading
import time
from collections import OrderedDict
from typing import Callable, Optional


# ---------- 1. 线程安全的 LRU ----------

class ThreadSafeLRU:
    """一把锁保护整个结构。get 也要加锁：它会改链表顺序。"""

    def __init__(self, capacity: int):
        self.cap = capacity
        self.od: "OrderedDict[int, int]" = OrderedDict()
        self.lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key: int) -> Optional[int]:
        with self.lock:
            if key not in self.od:
                self.misses += 1
                return None
            self.od.move_to_end(key)
            self.hits += 1
            return self.od[key]

    def put(self, key: int, value: int) -> None:
        with self.lock:
            if key in self.od:
                self.od.move_to_end(key)
            self.od[key] = value
            if len(self.od) > self.cap:
                self.od.popitem(last=False)

    def get_or_compute(self, key: int, compute: Callable[[int], int]) -> int:
        """缓存穿透的经典陷阱：不能在持锁时算 compute（慢），也不能松锁后不再检查（重复计算）。
        这里用"锁外计算、锁内二次检查"。"""
        v = self.get(key)
        if v is not None:
            return v
        v = compute(key)
        with self.lock:
            if key in self.od:                                   # 别人已经放进来了：用别人的
                self.od.move_to_end(key)
                return self.od[key]
            self.od[key] = v
            if len(self.od) > self.cap:
                self.od.popitem(last=False)
            return v


# ---------- 2. 有界阻塞队列 + 生产者消费者 ----------

class BoundedQueue:
    """用一把锁 + 两个条件变量手写（不用 queue.Queue）。"""

    def __init__(self, capacity: int):
        self.cap = capacity
        self.items: list = []
        self.lock = threading.Lock()
        self.not_full = threading.Condition(self.lock)
        self.not_empty = threading.Condition(self.lock)

    def put(self, item) -> None:
        with self.not_full:
            while len(self.items) >= self.cap:                   # while 而不是 if：防虚假唤醒与竞争
                self.not_full.wait()
            self.items.append(item)
            self.not_empty.notify()

    def get(self):
        with self.not_empty:
            while not self.items:
                self.not_empty.wait()
            item = self.items.pop(0)
            self.not_full.notify()
            return item


def producer_consumer_demo(n_items: int, n_producers: int, n_consumers: int, cap: int) -> int:
    q = BoundedQueue(cap)
    total = [0]
    total_lock = threading.Lock()
    SENTINEL = object()

    def producer(pid):
        for i in range(n_items):
            q.put(pid * n_items + i)

    def consumer():
        while True:
            x = q.get()
            if x is SENTINEL:
                return
            with total_lock:
                total[0] += x

    ps = [threading.Thread(target=producer, args=(p,)) for p in range(n_producers)]
    cs = [threading.Thread(target=consumer) for _ in range(n_consumers)]
    for t in ps + cs:
        t.start()
    for t in ps:
        t.join()
    for _ in cs:
        q.put(SENTINEL)                                          # 每个消费者一个哨兵
    for t in cs:
        t.join()
    return total[0]


# ---------- 3. 线程池 ----------

class ThreadPool:
    """固定 n 个 worker 从任务队列取任务；submit 返回一个简易 Future。"""

    class Future:
        def __init__(self):
            self._done = threading.Event()
            self._result = None
            self._exc: Optional[BaseException] = None

        def set(self, result=None, exc=None):
            self._result, self._exc = result, exc
            self._done.set()

        def result(self, timeout=None):
            self._done.wait(timeout)
            if self._exc:
                raise self._exc
            return self._result

    def __init__(self, n_workers: int):
        self.tasks: "queue.Queue" = queue.Queue()
        self.workers = [threading.Thread(target=self._run, daemon=True) for _ in range(n_workers)]
        for w in self.workers:
            w.start()

    def _run(self):
        while True:
            item = self.tasks.get()
            if item is None:
                return
            fn, args, fut = item
            try:
                fut.set(result=fn(*args))
            except BaseException as e:                           # 异常要传回 Future，不能让 worker 死掉
                fut.set(exc=e)

    def submit(self, fn, *args) -> "ThreadPool.Future":
        fut = ThreadPool.Future()
        self.tasks.put((fn, args, fut))
        return fut

    def shutdown(self):
        for _ in self.workers:
            self.tasks.put(None)
        for w in self.workers:
            w.join()


# ---------- 4. ring allreduce 模拟 ----------

def ring_allreduce(chunks_per_rank: list[list[float]]) -> tuple[list[list[float]], int]:
    """N 个 rank，每个 rank 持有长度 N 的向量（已切成 N 块）。
    reduce-scatter N-1 步 + all-gather N-1 步；每步每个 rank 发送一块。返回 (结果, 每 rank 发送的块数)。"""
    N = len(chunks_per_rank)
    data = [list(c) for c in chunks_per_rank]
    sent = 0
    for step in range(N - 1):                                    # reduce-scatter
        new = [list(d) for d in data]
        for r in range(N):
            idx = (r - step) % N                                 # rank r 在第 step 步发块 idx 给右邻
            new[(r + 1) % N][idx] += data[r][idx]
            sent += 1
        data = new
    # 此时 rank r 持有块 (r+1)%N 的完整和
    for step in range(N - 1):                                    # all-gather
        new = [list(d) for d in data]
        for r in range(N):
            idx = (r + 1 - step) % N
            new[(r + 1) % N][idx] = data[r][idx]
            sent += 1
        data = new
    return data, sent // N


# ---------- 5. paged KV block 分配器 ----------

class BlockAllocator:
    """vLLM 风格：显存切成固定大小的 block，序列按需申请，用引用计数支持前缀共享 / copy-on-write。"""

    def __init__(self, n_blocks: int):
        self.free: list[int] = list(range(n_blocks))
        self.ref: list[int] = [0] * n_blocks

    def alloc(self) -> int:
        if not self.free:
            raise MemoryError("out of KV blocks")
        b = self.free.pop()
        self.ref[b] = 1
        return b

    def share(self, b: int) -> int:
        self.ref[b] += 1
        return b

    def release(self, b: int) -> None:
        self.ref[b] -= 1
        if self.ref[b] == 0:
            self.free.append(b)

    def copy_on_write(self, b: int) -> int:
        """要写一个共享块：引用数 > 1 就先复制。"""
        if self.ref[b] == 1:
            return b
        self.ref[b] -= 1
        return self.alloc()

    @property
    def n_free(self) -> int:
        return len(self.free)


class Sequence:
    def __init__(self, allocator: BlockAllocator, block_size: int):
        self.alloc, self.bs = allocator, block_size
        self.blocks: list[int] = []
        self.n_tokens = 0

    def append_token(self):
        if self.n_tokens % self.bs == 0:                         # 当前块满了，要新块
            self.blocks.append(self.alloc.alloc())
        elif self.alloc.ref[self.blocks[-1]] > 1:                # 最后一块是共享的：写前复制
            self.blocks[-1] = self.alloc.copy_on_write(self.blocks[-1])
        self.n_tokens += 1

    def fork(self) -> "Sequence":
        """共享全部已有块（beam search / 多采样的前缀共享）。"""
        child = Sequence(self.alloc, self.bs)
        child.blocks = [self.alloc.share(b) for b in self.blocks]
        child.n_tokens = self.n_tokens
        return child

    def free(self):
        for b in self.blocks:
            self.alloc.release(b)
        self.blocks = []


# ---------- 6. token bucket 限流 ----------

class TokenBucket:
    """容量 cap、每秒补 rate 个令牌；允许突发到 cap。懒补充：每次请求时按流逝时间补。"""

    def __init__(self, rate: float, capacity: int, now: Callable[[], float] = time.monotonic):
        self.rate, self.cap, self.now = rate, capacity, now
        self.tokens = float(capacity)
        self.last = now()
        self.lock = threading.Lock()

    def try_acquire(self, n: int = 1) -> bool:
        with self.lock:
            t = self.now()
            self.tokens = min(self.cap, self.tokens + (t - self.last) * self.rate)
            self.last = t
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False


# ---------- self-check ----------

def main():
    # 1 线程安全 LRU：多线程并发 get_or_compute，容量 8，key 空间 16
    lru = ThreadSafeLRU(8)
    computed = [0]
    clock = threading.Lock()

    def compute(k):
        with clock:
            computed[0] += 1
        time.sleep(0.0005)
        return k * k

    def worker(seed):
        for i in range(200):
            k = i % 6 if i % 5 else (seed * 7 + i) % 16      # 热点 6 个 key + 少量冷 key
            assert lru.get_or_compute(k, compute) == k * k
    ts = [threading.Thread(target=worker, args=(s,)) for s in range(8)]
    for t in ts: t.start()
    for t in ts: t.join()
    assert len(lru.od) <= 8
    print(f"1 LRU: 1600 次访问，命中 {lru.hits}，未命中 {lru.misses}，实际计算 {computed[0]} 次，表大小 {len(lru.od)}")

    # 2 生产者消费者
    total = producer_consumer_demo(n_items=500, n_producers=3, n_consumers=2, cap=8)
    expect = sum(range(3 * 500))
    assert total == expect, (total, expect)
    print(f"2 有界队列（容量 8）：3 生产者 × 500 → 2 消费者，和 = {total} ✓")

    # 3 线程池
    pool = ThreadPool(4)
    futs = [pool.submit(lambda x: x * x, i) for i in range(20)]
    bad = pool.submit(lambda: 1 / 0)
    assert [f.result() for f in futs] == [i * i for i in range(20)]
    try:
        bad.result()
        raise AssertionError("should raise")
    except ZeroDivisionError:
        pass
    pool.shutdown()
    print("3 线程池：20 个任务结果正确，异常正确传回 Future ✓")

    # 4 ring allreduce
    N = 4
    ranks = [[float(r * 10 + i) for i in range(N)] for r in range(N)]
    out, per_rank_sent = ring_allreduce(ranks)
    expect_vec = [sum(ranks[r][i] for r in range(N)) for i in range(N)]
    assert all(o == expect_vec for o in out)
    print(f"4 ring allreduce（N={N}）：每 rank 发送 {per_rank_sent} 块 = 2(N−1)；总量 2(N−1)/N ≈ {2 * (N - 1) / N:.2f} 倍向量大小，与 N 无关 ✓")

    # 5 paged KV
    alloc = BlockAllocator(n_blocks=8)
    s = Sequence(alloc, block_size=4)
    for _ in range(6):
        s.append_token()                                         # 2 块（4 + 2）
    child = s.fork()                                             # 共享 2 块
    assert alloc.n_free == 6 and alloc.ref[s.blocks[1]] == 2
    child.append_token()                                         # 最后一块共享 → copy-on-write
    assert alloc.n_free == 5 and alloc.ref[s.blocks[1]] == 1 and child.blocks[1] != s.blocks[1]
    s.free(); child.free()
    assert alloc.n_free == 8
    print("5 paged KV：6 token 占 2 块；fork 共享（引用计数 2）；子序列追加触发 COW 多用 1 块；释放后全部回收 ✓")

    # 6 token bucket（用假时钟）
    fake = [0.0]
    tb = TokenBucket(rate=10, capacity=5, now=lambda: fake[0])
    burst = sum(tb.try_acquire() for _ in range(10))
    fake[0] += 0.25                                              # 补 2.5 个
    after = sum(tb.try_acquire() for _ in range(10))
    assert burst == 5 and after == 2
    print(f"6 token bucket（rate 10/s, cap 5）：突发通过 {burst}，0.25 s 后再通过 {after} ✓")


if __name__ == "__main__":
    main()
