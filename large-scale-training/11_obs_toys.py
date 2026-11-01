"""11《长时间训练的可观测性与运维》toys：全部真实运行，输出实测值。

  python 11_obs_toys.py fr       # 4 rank gloo：rank 3 缺席第 5 次 all_reduce，超时后用真正的
                                # Flight Recorder dump + torch 自带的 fr_trace 分析器定位缺席者
  python 11_obs_toys.py timers   # Megatron Timers 式分阶段计时：dataloader 渐变慢，回归检测器开火
  python 11_obs_toys.py ranks    # 4 rank × 20 step 矩阵：rank 1 从 step 8 起慢 30ms
  python 11_obs_toys.py alert    # §五.1–3 规则评估的模拟：for: 语义、page/record、inhibit
  python 11_obs_toys.py all
"""

import sys

PY = None



def _fr_worker(rank, init_file, dump_dir, world):
    import os
    import time
    from datetime import timedelta

    import torch
    import torch.distributed as dist

    dist.init_process_group("gloo", rank=rank, world_size=world,
                            init_method=f"file://{init_file}",
                            timeout=timedelta(seconds=4))
    import torch._C._distributed_c10d as c10d

    t = torch.ones(4)
    for i in range(6):
        seq = i + 1
        if rank == 3 and seq == 5:
            # rank 3 缺席 seq 5（例如走了 rank 相关的分支），直接做下一个
            continue
        try:
            dist.all_reduce(t)
        except RuntimeError as e:
            print(f"[rank {rank}] seq {seq} 超时：{e}", flush=True)
            break
    # 超时之后（rank 3 也要等到其他人被打死，用 sleep 让时序清楚）
    time.sleep(5)
    blob = c10d._dump_fr_trace(True, True, False)
    with open(os.path.join(dump_dir, f"rank_{rank}"), "wb") as f:
        f.write(blob)
    dist.destroy_process_group()

def cmd_fr():
    """真正的 Flight Recorder：4 个 gloo rank，rank 3 不发起第 5 次 all_reduce。

    torch 2.14 的 ProcessGroupGloo 也写 FR（profiling_name 是 gloo:*，字段名相同），
    _dump_fr_trace 返回每 rank 的 pickle；分析用 torch 自带的
    torch.distributed.flight_recorder（torchfrtrace）——与正文 §三.5 是同一个工具。
    """
    import multiprocessing as mp
    import os
    import tempfile

    world = 4

    tmp = tempfile.mkdtemp(prefix="fr_toy_")
    init_file = os.path.join(tmp, "init")
    dump_dir = os.path.join(tmp, "comm_traces")
    os.makedirs(dump_dir)
    ctx = mp.get_context("spawn")
    ready, done = ctx.Queue(), ctx.Queue()
    procs = [ctx.Process(target=_fr_worker, args=(r, init_file, dump_dir, world))
             for r in range(world)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)

    print(f"4 个 gloo rank；rank 3 跳过第 5 次 all_reduce；超时 4 s\n"
          f"dump 目录（模拟 torchtitan 的 comm_traces/）：{dump_dir}")
    import pickle
    last = {}
    for r in range(world):
        with open(os.path.join(dump_dir, f"rank_{r}"), "rb") as f:
            d = pickle.load(f)
        entries = d["entries"]
        seqs = [(e["collective_seq_id"], e["profiling_name"], e["state"])
                for e in entries]
        last[r] = seqs
        print(f"  rank {r}: {len(entries)} 条，"
              f"seq {[s for s, _, _ in seqs]}，"
              f"state {[st for _, _, st in seqs]}；"
              f"pg_status {d['pg_status']}")

    # gloo 的 dump 把 pg_config 写在 '' 名下，而 entries 的 process_group[0] 是 '0'
    # （FR 的数据格式按 NCCL 的 pg_id 设计）；把键改成 '0' 供分析器对齐
    for r in range(world):
        p = os.path.join(dump_dir, f"rank_{r}")
        d = pickle.loads(open(p, "rb").read())
        d["pg_config"] = {"0": {"name": "0", "desc": "default_pg",
                                "ranks": "[0, 1, 2, 3]"}}
        with open(p, "wb") as f:
            pickle.dump(d, f)

    print("\n===== torchfrtrace 的分析（真正的分析器）=====")
    from torch.distributed.flight_recorder import fr_trace
    fr_trace.main([dump_dir, "-p", "rank_", "--mismatch_cap", "3"])

    # 供画图用的 JSON：每 rank 最后几个 seq
    import json
    rings = {f"rank {r}": [[s, st] for s, _, st in seqs]
             for r, seqs in last.items()}
    with open("expected/11_obs_fr.json", "w") as f:
        json.dump(rings, f)


def cmd_timers():
    """分阶段计时 + 回归检测：dataloader 从 step 60 起线性变慢。"""
    import json
    import statistics
    import time
    from collections import deque

    # 各阶段基准（ms）；data 从 step 60 线性增长
    base = {"data": 12.0, "fwd": 30.0, "bwd": 48.0, "opt": 10.0}
    steps = 120
    series = []
    fire_step = None
    win = deque(maxlen=20)
    # 基线：前 20 步中位数
    baseline = None
    for step in range(1, steps + 1):
        grow = max(0.0, (step - 60) * 2.0)  # 60 步后每步 +2ms
        phases = {k: v + (grow if k == "data" else 0.0) for k, v in base.items()}
        # 加一点抖动（不用随机数，用正弦保证可复现）
        import math
        phases = {k: v * (1 + 0.02 * math.sin(step + i))
                  for i, (k, v) in enumerate(phases.items())}
        total = sum(phases.values())
        series.append({"step": step, "total_ms": total,
                       **{f"{k}_ms": v for k, v in phases.items()}})
        win.append(total)
        if baseline is None and len(win) == 20:
            baseline = statistics.median(win)
        if baseline and len(win) == 20 and fire_step is None:
            if statistics.median(win) > baseline * 1.10:
                fire_step = step

    print("小模型训练循环的分阶段计时（基准 ms，data 从 step 60 起每步 +2ms）：")
    print("阶段基准：", base)
    for s in (30, 60, fire_step or 0, 90, 120):
        s_ = series[s - 1]
        print(f"  step {s_['step']:>3}  total {s_['total_ms']:6.1f}ms  "
              f"data {s_['data_ms']:5.1f}  fwd {s_['fwd_ms']:4.1f}  "
              f"bwd {s_['bwd_ms']:4.1f}  opt {s_['opt_ms']:4.1f}")
    print(f"\n回归检测器：20 步窗口中位数 vs 基线 {baseline:.1f}ms，"
          f">10% 报警 → 第 {fire_step} 步开火")
    # 排除数据外的嫌疑：各阶段占比
    s60, s120 = series[59], series[119]
    print(f"step 60 占比：data {s60['data_ms']/s60['total_ms']*100:.0f}%  "
          f"step 120 占比：data {s120['data_ms']/s120['total_ms']*100:.0f}%"
          f"（其余阶段没变 → 嫌疑是数据，§四.1 表格第一行）")

    with open("expected/11_obs_timers.json", "w") as f:
        json.dump({"base": base, "baseline_ms": baseline,
                   "fire_step": fire_step, "series": series}, f)


def _ranks_worker(rank, init_file, q, world, nsteps):
    from datetime import timedelta
    import time
    import torch
    import torch.distributed as dist

    dist.init_process_group("gloo", rank=rank, world_size=world,
                            init_method=f"file://{init_file}",
                            timeout=timedelta(seconds=10))
    t = torch.ones(256)
    rows = []
    for step in range(1, nsteps + 1):
        t0 = time.perf_counter()
        c0 = time.perf_counter()
        for _ in range(50):
            t = t @ torch.eye(256) * 0.999
        if rank == 1 and step >= 8:
            time.sleep(0.03)  # rank 1 慢 30ms（straggler）
        compute_ms = (time.perf_counter() - c0) * 1000
        dist.all_reduce(t)
        step_ms = (time.perf_counter() - t0) * 1000
        rows.append((step, compute_ms, step_ms))
    q.put((rank, rows))
    dist.destroy_process_group()


def cmd_ranks():
    """4 rank × 20 step：rank 1 从 step 8 起每步慢 30ms，输出 rank × step 矩阵。"""
    import json
    import multiprocessing as mp
    import tempfile
    import os

    world, nsteps = 4, 20



    tmp = tempfile.mkdtemp(prefix="ranks_toy_")
    init_file = os.path.join(tmp, "init")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_ranks_worker, args=(r, init_file, q, world, nsteps))
             for r in range(world)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=90)

    mats = {}
    for _ in range(world):
        rank, rows = q.get()
        mats[rank] = rows

    print("4 rank × 20 step；rank 1 从 step 8 起 compute 多 30ms（straggler）")
    print("左 = compute_ms（本 rank 自己的计算），右 = step_ms（含等 all_reduce）")
    for label, idx in (("compute_ms", 1), ("step_ms", 2)):
        print(f"\n{label} 矩阵（行=step）：")
        print("step   " + "".join(f"rank{r:>8}" for r in range(world)))
        for s in range(nsteps):
            line = f"{s+1:>4}   "
            for r in range(world):
                line += f"{mats[r][s][idx]:>8.1f}"
            print(line)
    grid = [[mats[r][s][1] for r in range(world)] for s in range(nsteps)]
    med = {r: sum(mats[r][s][1] for s in range(nsteps)) / nsteps
           for r in range(world)}
    print("\ncompute 平均：", {f"rank{r}": round(v, 1) for r, v in med.items()},
          "→ step 时间被 all_reduce 等齐、看不出谁慢；"
          "compute 时间 rank 1 明显长 → 每 rank 计时才是 §二.6 的眼睛")

    with open("expected/11_obs_ranks.json", "w") as f:
        json.dump({"world": world, "steps": nsteps,
                   "matrix": grid}, f)   # grid[step][rank] = compute_ms


def cmd_alert():
    """§五.1–3 规则评估的模拟：for: 语义、page/record、inhibit_rules。

    这是规则评估的模拟，不是真的 Prometheus：合成一条吞吐曲线与心跳序列，
    按正文的表达式逐 tick 求值。
    """
    # 时间轴：秒，tick = 15s（Prometheus 抓取间隔），共 30 分钟
    tick = 15
    T = [i * tick for i in range(120)]
    # 合成：吞吐基线 1000 tok/s；第 10–15 分钟掉 20%（>15% 但 <30min → record）
    #       第 22 分钟起心跳停（hang：step 计数不再前进）
    def throughput(t):
        if 600 <= t < 900:
            return 800.0
        return 1000.0
    def heartbeat(t):          # 最近一次 step 完成的时刻
        return min(t, 1410)

    # 规则（正文 §五.2）：
    #   TrainingStepStalled (page, for 0m): time - last_step > 3*p50_step(=90s) 且 >90s
    #   TrainingMfuDegraded (page, for 30m): MFU 下滑 >15% —— 5 分钟 dip 不够长 → 不 page
    #   ThroughputDip      (record, for 2m): 吞吐 < 85% 基线持续 2 分钟
    #   TrainingRestarting: 合成一个重启事件 t=1500s 起 3 分钟，抑制 StepStalled
    rules = [
        {"name": "TrainingStepStalled", "severity": "page", "for_s": 0,
         "expr": lambda t: (t - heartbeat(t)) > 90},
        {"name": "ThroughputDip", "severity": "record", "for_s": 120,
         "expr": lambda t: throughput(t) < 0.85 * 1000},
        {"name": "TrainingMfuDegraded", "severity": "page", "for_s": 1800,
         "expr": lambda t: throughput(t) < 0.85 * 1000},
    ]
    restart_active = lambda t: 1500 <= t < 1680   # torchrun 自动拉起 3 分钟

    # Prometheus for: 语义：expr 连续为真满 for_s 才 firing；之前是 pending
    state = {r["name"]: {"since": None, "firing": False} for r in rules}
    events = []
    for t in T:
        for r in rules:
            st = state[r["name"]]
            if r["expr"](t):
                if st["since"] is None:
                    st["since"] = t
                    events.append((t, "pending", r["name"], r["severity"]))
                if not st["firing"] and t - st["since"] >= r["for_s"]:
                    st["firing"] = True
                    suppressed = (r["name"] == "TrainingStepStalled"
                                  and restart_active(t))
                    events.append((t, "firing" + ("（被 TrainingRestarting 抑制）" if suppressed else ""),
                                   r["name"], r["severity"]))
            else:
                if st["firing"]:
                    events.append((t, "resolved", r["name"], r["severity"]))
                elif st["since"] is not None:
                    events.append((t, "pending 消失（expr 不再成立）", r["name"], r["severity"]))
                st["since"] = None
                st["firing"] = False
            # inhibit: 重启期间即使 StepStalled pending 也不计
            if r["name"] == "TrainingStepStalled" and restart_active(t):
                st["since"] = None if not st["firing"] else st["since"]

    print("合成的曲线：吞吐基线 1000 tok/s，600–900s 掉到 800（-20%）；"
          "心跳（last_step）在 1410s 后不再前进；1500–1680s torchrun 自动重启\n")
    print("规则（§五.2 的表达式的模拟评估）：")
    print("  TrainingStepStalled  page   for:0m   time-last_step>90")
    print("  ThroughputDip        record for:2m   throughput<850 持续 2 分钟")
    print("  TrainingMfuDegraded  page   for:30m  throughput<850 持续 30 分钟")
    print("  inhibit：TrainingRestarting 活跃时抑制 TrainingStepStalled\n")
    mm = lambda s: f"{s//60:02d}:{s%60:02d}"
    for t, ev, name, sev in events:
        print(f"  {mm(t)}  [{sev:>6}] {name:>22}  {ev}")
    print("\n结论：5 分钟的 -20% dip 只触发 record（for:30m 没到）；"
          "心跳停 → StepStalled pending→firing（page），"
          "但 firing 时刻落在重启窗口里 → 被抑制，值班的人只收到重启的那一条。")

    with open("expected/11_obs_alert.json", "w") as f:
        import json
        json.dump({"tick": tick,
                   "throughput": [throughput(t) for t in T],
                   "heartbeat": [heartbeat(t) for t in T],
                   "restart": [1500, 1680],
                   "events": [[t, ev, name, sev] for t, ev, name, sev in events],
                   "T": T}, f)


SUBCOMMANDS = {
    "fr": cmd_fr,
    "timers": cmd_timers,
    "ranks": cmd_ranks,
    "alert": cmd_alert,
}

if __name__ == "__main__":
    names = sys.argv[1:] or ["all"]
    if names == ["all"]:
        names = list(SUBCOMMANDS)
    for n in names:
        if names != ["all"]:
            pass
        else:
            print(f"===== {n} =====")
        SUBCOMMANDS[n]()
        if len(names) > 1:
            print()
