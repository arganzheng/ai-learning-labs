"""Python 在 AI-Infra（03）：进程池 —— cancel 语义、contextvars 不跨进程、worker 崩溃 -> BrokenProcessPool、有界 mp.Queue。
https://arganzheng.life/python-concurrency-asynchrony-and-task-collaboration.html
"""
import os, sys, time, multiprocessing as mp, contextvars
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FTimeout
from concurrent.futures.process import BrokenProcessPool

cv = contextvars.ContextVar("rid", default="-")

def slow(n): time.sleep(0.3); return n
def show_ctx(_): return (os.getpid(), cv.get())
def crash(_): os._exit(1)      # 模拟 worker 被 OOM killer 干掉

if __name__ == "__main__":
    print("start method:", mp.get_start_method(), "| main pid", os.getpid())
    # 1. Future.cancel：同线程池，只对未开始的有效；result(timeout) 不停止 worker
    with ProcessPoolExecutor(max_workers=1) as ex:
        f1 = ex.submit(slow, 1); f2 = ex.submit(slow, 2); time.sleep(0.1)
        print("cancel running:", f1.cancel(), "| cancel pending:", f2.cancel())
    # 2. contextvars 不跨进程
    cv.set("req-42")
    with ProcessPoolExecutor(max_workers=1) as ex:
        print("in worker:", ex.submit(show_ctx, None).result(), "| in main:", cv.get())
    # 3. worker 崩溃 -> BrokenProcessPool
    with ProcessPoolExecutor(max_workers=1) as ex:
        f = ex.submit(crash, None)
        try: f.result()
        except BrokenProcessPool as e: print("BrokenProcessPool:", str(e)[:70])
    # 4. mp.Queue 有界
    q = mp.Queue(maxsize=2); q.put(1); q.put(2)
    try: q.put(3, timeout=0.05)
    except Exception as e: print("mp.Queue full ->", type(e).__module__ + "." + type(e).__name__)
