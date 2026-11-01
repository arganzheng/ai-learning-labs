"""《容错与弹性》toys：mtbf / watchdog / elastic / straggler。

CPU + gloo 真实多进程（_spawn 起 N 个 __worker__ 子进程）。
数字全部来自正文表格与推导，只打印实测，不凑数。
"""
import os
import subprocess
import sys
import tempfile
import time

PORT = "29519"


def _spawn(fn, world, tmpdir, extra_args=None, stagger=0.0, port=None):
    args = [sys.executable, __file__, "__worker__", fn, str(world), tmpdir] \
        + (extra_args or [])
    env = dict(os.environ)
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = port or PORT
    procs = []
    for r in range(world):
        procs.append(subprocess.Popen(args + [str(r)], env=env))
        time.sleep(stagger)
    for p in procs:
        p.wait()
    if any(p.returncode for p in procs):
        raise SystemExit(f"worker failed: {[p.returncode for p in procs]}")


# ---------- mtbf：集群 MTBF 与五项有效时间（§二.1 / §二.4 表） ----------

def _G(delta, M, Td, Tr, Tl):
    t_opt = (2 * delta * M) ** 0.5
    L = Td + Tr + Tl + t_opt / 2
    g = (1 - L / M) / (1 + delta / t_opt)
    parts = {
        "检测 T_d": Td / M,
        "重启 T_r": Tr / M,
        "加载 T_l": Tl / M,
        "回退 τ/2": t_opt / 2 / M,
        "保存 δ/τ": (delta / t_opt) * g,  # 分母项折算进 G 的份额
    }
    return t_opt, g, parts


def cmd_mtbf(_):
    M_GPU = 50000.0  # h，Llama 3 反推（正文 §一.2 表末行 ≈5.1 万小时）
    print(f"单卡 M_gpu = {M_GPU:.0f} h（≈{M_GPU/24/365:.1f} 年）")
    for N in (1024, 16384):
        print(f"  N={N:>6}: 集群 M = M_gpu/N = {M_GPU/N:6.2f} h = {M_GPU/N*3600:8.0f} s")
    print()
    scenes = {
        "A 同步 δ=300s Td=600s Tr=300s Tl=120s": (300, 600, 300, 120),
        "B 异步 δ=10s  其余同 A            ": (10, 600, 300, 120),
        "C 异步 δ=10s  Td=60s Tr=30s Tl=60s": (10, 60, 30, 60),
    }
    for name, (d, td, tr, tl) in scenes.items():
        print(f"-- 场景 {name.strip()}")
        for N in (1024, 16384):
            M = M_GPU / N * 3600
            t_opt, g, parts = _G(d, M, td, tr, tl)
            seg = "、".join(f"{k} {v*100:.1f}%" for k, v in parts.items())
            print(f"   N={N:>6} (M={M/3600:5.1f} h): τ_opt={t_opt/60:5.1f} min"
                  f"  G={g*100:.1f}%  [{seg}]")
    print("\n五项分解（16K 卡，G 的构成，与 F-new1 一致）：")
    for name, (d, td, tr, tl) in scenes.items():
        M = M_GPU / 16384 * 3600
        t_opt, g, parts = _G(d, M, td, tr, tl)
        tot = (Td := parts)
        loss = sum(v for k, v in tot.items() if k != "保存 δ/τ")
        print(f"   {name.split()[0]}: 有效 {g*100:.1f}% + "
              + " ".join(f"{k.split()[-1]} {v*100:.1f}%" for k, v in tot.items()))


# ---------- watchdog：2 rank，一个故意 hang（§三.3 的 gloo 类比） ----------

def _worker_watchdog(world, rank, tmpdir):
    import datetime
    import torch
    import torch.distributed as dist
    # store 用默认超时（集合通信的 3s 超时才由 pg timeout 控制）
    store = dist.TCPStore("127.0.0.1", int(PORT), world, rank == 0,
                          datetime.timedelta(seconds=60))
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=3))
    x = torch.ones(4)
    if rank == 0:
        t0 = time.monotonic()
        try:
            dist.all_reduce(x)
            print("rank 0: all_reduce 意外成功", flush=True)
        except Exception as e:
            el = time.monotonic() - t0
            print(f"rank 0: {el:.1f}s 后捕获 {type(e).__name__}: {e}", flush=True)
    else:
        print("rank 1: 睡 6s 模拟 hang（不进 all_reduce）", flush=True)
        time.sleep(6)
        print("rank 1: 醒来，此时 rank 0 已超时", flush=True)
    dist.destroy_process_group()


# ---------- elastic：torchft 副本组机制的多进程模拟（§六） ----------

def _worker_elastic(world, rank, tmpdir):
    # 副本间协调用 tmpdir 里的文件（模拟 torchft 的 store / transport，见注记）
    def put(key, val):
        p = os.path.join(tmpdir, key)
        with open(p, "w") as f:
            f.write(str(val))

    def get(key, deadline=60.0):
        p = os.path.join(tmpdir, key)
        t0 = time.monotonic()
        while not os.path.exists(p):
            if time.monotonic() - t0 > deadline:
                raise TimeoutError(key)
            time.sleep(0.01)
        return float(open(p).read())

    lr = 0.1
    w = 0.0  # 副本从同一参数出发；rank 1 掉线期间落后，heal 后应对齐
    for step in range(6):
        live = [r for r in range(world) if not (r == 1 and step in {3, 4})]
        if rank == 1 and step == 5:
            w = get("w_0")  # heal：从邻居拉参数（§六.4）
            print(f"rank 1: step 5 heal，从 rank 0 拉参数 w={w:.4f}", flush=True)
        if rank in live:
            g = w + 1.0 + rank  # 各副本本地梯度不同，看归一化
            put(f"g_{step}_{rank}", g)
        grads = [get(f"g_{step}_{r}") for r in live]
        if rank in live:
            w -= lr * sum(grads) / len(grads)  # 按 quorum 大小归一化
        if step == 4 and rank == 0:
            put("w_0", w)  # 供 rank 1 下一步 heal
        if rank == 0:
            print(f"step {step}: quorum={live}  rank0 w={w:.4f}", flush=True)
    put(f"final_{rank}", w)
    if rank == 0:
        vals = [get(f"final_{r}") for r in range(world)]
        print(f"收尾：三副本 w = {[f'{v:.4f}' for v in vals]}"
              f"  最大差 = {max(vals)-min(vals):.2e}", flush=True)


# ---------- straggler：4 rank，rank 2 每步多睡 50ms（§八.2） ----------

def _worker_straggler(world, rank, tmpdir):
    import torch
    import torch.distributed as dist
    dist.init_process_group("gloo", rank=rank, world_size=world)
    x = torch.ones(8)
    comp = wait = 0.0
    for _ in range(3):
        t0 = time.monotonic()
        _ = (x * x).sum()  # 一点点计算
        time.sleep(0.05 if rank == 2 else 0.0)
        comp += time.monotonic() - t0
        t0 = time.monotonic()
        dist.all_reduce(x)
        wait += time.monotonic() - t0
    print(f"rank {rank}: 计算 {comp*1000:6.1f} ms   all_reduce 内等待 {wait*1000:6.1f} ms",
          flush=True)
    dist.destroy_process_group()


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: mtbf|watchdog|elastic|straggler|all")
    a = sys.argv[1:]
    if a[0] == "__worker__":
        fn, world, tmpdir, rank = a[1], int(a[2]), a[3], int(a[4])
        {"watchdog": _worker_watchdog,
         "elastic": _worker_elastic,
         "straggler": _worker_straggler}[fn](world, rank, tmpdir)
        return
    cmds = a if a[0] != "all" else ["mtbf", "watchdog", "elastic", "straggler"]
    for c in cmds:
        if c == "mtbf":
            print("===== mtbf =====")
            cmd_mtbf(None)
        elif c == "watchdog":
            print("===== watchdog（2 rank，timeout=3s）=====")
            _spawn("watchdog", 2, tempfile.mkdtemp(), stagger=1.5, port="29519")
        elif c == "elastic":
            print("===== elastic（3 副本模拟 torchft）=====")
            _spawn("elastic", 3, tempfile.mkdtemp())
        elif c == "straggler":
            print("===== straggler（4 rank，rank 2 慢 50ms/步）=====")
            _spawn("straggler", 4, tempfile.mkdtemp(), port="29520")


if __name__ == "__main__":
    main()
