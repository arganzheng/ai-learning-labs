"""Python 在 AI-Infra（03）：线程池 —— Future.cancel 的范围、result(timeout) 不停 worker、shutdown(cancel_futures)、Event 停止标志、threading.local vs contextvars、有界队列。
https://arganzheng.life/python-concurrency-asynchrony-and-task-collaboration.html
"""
import time, threading, asyncio, contextvars, queue, sys
from concurrent.futures import ThreadPoolExecutor, wait

print(sys.version.split()[0])
# 1. Future.cancel only works for not-yet-started tasks
def slow(n): time.sleep(0.3); return n
with ThreadPoolExecutor(max_workers=1) as ex:
    f1 = ex.submit(slow, 1); f2 = ex.submit(slow, 2)
    time.sleep(0.05)
    print("cancel running:", f1.cancel(), "| cancel pending:", f2.cancel())
    print("f1.result:", f1.result(), "| f2.cancelled:", f2.cancelled())

# 2. result(timeout) does not stop the worker
with ThreadPoolExecutor(max_workers=1) as ex:
    t0 = time.perf_counter()
    f = ex.submit(slow, 3)
    try: f.result(timeout=0.05)
    except TimeoutError as e: print("TimeoutError after %.2fs, running=%s" % (time.perf_counter()-t0, f.running()))
print("exit with-block took %.2fs (waited for worker)" % (time.perf_counter()-t0))

# 3. shutdown(cancel_futures=True)
ex = ThreadPoolExecutor(max_workers=1)
fs = [ex.submit(slow, i) for i in range(4)]
time.sleep(0.05); ex.shutdown(cancel_futures=True)
print("states:", [("cancelled" if f.cancelled() else "done") for f in fs])

# 4. stop flag via Event
stop = threading.Event()
def worker():
    n = 0
    while not stop.is_set():
        n += 1; time.sleep(0.01)
    print("worker stopped after", n, "iterations")
t = threading.Thread(target=worker); t.start(); time.sleep(0.1); stop.set(); t.join()

# 5. threading.local vs contextvars in asyncio; to_thread copies context
tl = threading.local(); cv = contextvars.ContextVar("rid", default="-")
async def handler(rid):
    tl.rid = rid; cv.set(rid)
    await asyncio.sleep(0.01)
    in_thread = await asyncio.to_thread(lambda: (getattr(tl, "rid", None), cv.get()))
    print(f"task {rid}: local={tl.rid} ctxvar={cv.get()} | in to_thread: local={in_thread[0]} ctxvar={in_thread[1]}")
async def main(): await asyncio.gather(handler("A"), handler("B"))
asyncio.run(main())

# 6. queue.Queue maxsize
q = queue.Queue(maxsize=2); q.put(1); q.put(2)
try: q.put(3, timeout=0.05)
except queue.Full: print("queue.Full after timeout")
print("put_nowait ->", end=" ")
try: q.put_nowait(3)
except queue.Full: print("queue.Full")
