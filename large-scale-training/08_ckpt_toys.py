"""《分布式 checkpoint》小实验：真实 torch.distributed.checkpoint（CPU/gloo）。

每个子命令用 subprocess 起 world 个 gloo 进程。state_dict 里的张量做成
DTensor Shard(0)（等价于 FSDP 的分片方式）：一维参数 24 个元素，4 rank 各存
连续的 6 个；reshard 时换 3 rank / 2 rank 加载，验证重分片。

子命令：
  save     4 rank 存：打印文件清单与 .metadata 块表
  reshard  把 4-rank 的存盘加载进 3 rank 和 2 rank：每 rank 的本地块与误差
  async    dcp.save vs dcp.async_save 的阻塞时间对比（CPU 数字只看形状）
  young    Young 公式表：几个 δ 下的最优间隔与浪费（正文 §九.3 的数字）
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

NEL = 24          # 一个一维"参数"：24 个元素，4/3/2 都整除
NSTEPS = 3        # 优化器 m/v 与参数一样切法；另加标量 train_state


def _state_dict(world, mesh, fill=None):
    """FSDP 风格的 state_dict：参数与优化器 m/v 都是 Shard(0) DTensor。

    fill：用来填充参数的函数 i -> value；None 表示用保存时的公式。
    """
    f = fill or (lambda i: i * 0.5 + 1.0)
    full = torch.tensor([f(i) for i in range(NEL)], dtype=torch.float32)
    chunk = full.chunk(world)[dist.get_rank()]
    sd = {}
    for k in ("w", "opt.m", "opt.v"):
        scale = {"w": 1.0, "opt.m": 0.1, "opt.v": 0.01}[k]
        sd[k] = DTensor.from_local((full * scale).chunk(world)[dist.get_rank()],
                                   mesh, [Shard(0)])
    sd["train_state"] = torch.tensor([12345])      # 标量，每个 rank 一份
    return sd


def _worker_save(rank, world, tmpdir):
    dist.init_process_group("gloo", rank=rank, world_size=world)
    mesh = init_device_mesh("cpu", (world,))
    sd = _state_dict(world, mesh)
    ckpt_dir = os.path.join(tmpdir, "ckpt")
    dcp.save(sd, checkpoint_id=ckpt_dir)
    dist.barrier()
    if rank == 0:
        print(f"== dcp.save → {os.path.basename(ckpt_dir)}/（4 rank，每 rank 一个分片文件）")
        for f in sorted(os.listdir(ckpt_dir)):
            p = os.path.join(ckpt_dir, f)
            print(f"  {f}  {os.path.getsize(p)} B")
        import torch.distributed.checkpoint.filesystem as fs
        m = fs.FileSystemReader(ckpt_dir).read_metadata()
        print("\n== .metadata：state_dict_metadata（FQN → 全局形状/dtype）")
        for fqn, st in sorted(m.state_dict_metadata.items()):
            sz = getattr(st, "size", None)
            dt = getattr(st, "properties", None)
            print(f"  {fqn}  size={tuple(sz) if sz is not None else '-'}")
        print("\n== .metadata：storage_data（磁盘块 → 文件内偏移）")
        for idx, si in sorted(m.storage_data.items(), key=lambda kv: str(kv[0])):
            print(f"  {idx.fqn} chunk@{tuple(idx.offset)} -> {si.relative_path}"
                  f" off={si.offset} len={si.length}")
    dist.destroy_process_group()


def _worker_reshard(rank, world, tmpdir, src_dir):
    """把 world=4 存的 checkpoint 加载进本 world；打印本地块、读来源与误差。"""
    dist.init_process_group("gloo", rank=rank, world_size=world)
    mesh = init_device_mesh("cpu", (world,))
    sd = _state_dict(world, mesh, fill=lambda i: -999.0)   # 故意填脏数据
    # 记录加载前 rank 的本地区间
    lo, hi = rank * (NEL // world), (rank + 1) * (NEL // world)
    dcp.load(sd, checkpoint_id=src_dir)
    w = sd["w"].to_local()
    ref = torch.tensor([i * 0.5 + 1.0 for i in range(NEL)], dtype=torch.float32)
    err = (w - ref[lo:hi]).abs().max().item()
    m_ok = (sd["opt.m"].to_local() - ref[lo:hi] * 0.1).abs().max().item()
    v_ok = (sd["opt.v"].to_local() - ref[lo:hi] * 0.01).abs().max().item()
    print(f"  rank {rank}: 本地块 w[{lo}:{hi}] = {[f'{x:.1f}' for x in w.tolist()]}"
          f"  误差 w={err:.1e} m={m_ok:.1e} v={v_ok:.1e}")
    dist.destroy_process_group()


def _worker_async(rank, world, tmpdir):
    import time
    dist.init_process_group("gloo", rank=rank, world_size=world)
    mesh = init_device_mesh("cpu", (world,))
    big = torch.randn(2_000_000)   # ~8 MB，放大到计时可见
    sd = {"big": DTensor.from_local(
        big.chunk(world)[rank].clone().contiguous(), mesh, [Shard(0)])}
    t0 = time.perf_counter()
    dcp.save(sd, checkpoint_id=os.path.join(tmpdir, "sync"))
    t_sync = time.perf_counter() - t0
    t0 = time.perf_counter()
    fut = dcp.async_save(sd, checkpoint_id=os.path.join(tmpdir, "async"))
    t_stage = time.perf_counter() - t0
    fut.result()
    t_total = time.perf_counter() - t0
    if rank == 0:
        print("== async 计时（CPU/gloo，~8 MB 分片张量，数字只看形状不看大小）")
        print(f"  dcp.save       阻塞 {t_sync*1000:.1f} ms")
        print(f"  dcp.async_save 返回前 staging 阻塞 {t_stage*1000:.1f} ms，"
              f"后台写完共 {t_total*1000:.1f} ms")
    dist.destroy_process_group()


def _spawn(fn, world, tmpdir, extra_args=None):
    args = [sys.executable, __file__, "__worker__", fn, str(world), tmpdir] \
        + (extra_args or [])
    env = dict(os.environ)
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29517"
    procs = [subprocess.Popen(args + [str(r)], env=env) for r in range(world)]
    for p in procs:
        p.wait()
    if any(p.returncode for p in procs):
        raise SystemExit(f"worker failed: {[p.returncode for p in procs]}")


def cmd_young(_):
    import math
    M = 54 * 24 / 419
    print(f"M = 54*24/419 = {M:.2f} h（Llama 3，正文 §九.3）")
    print(f"{'δ':>8} {'τ_opt':>10} {'f_min':>7} {'期望损失/故障':>13} {'每天保存':>9}  实现")
    for delta, impl in [(600, "同步单写入者"), (120, "同步分片写、慢 PFS"),
                        (60, "同步分片写"), (10, "异步 staging"), (2, "异步 pinned+重叠")]:
        t = math.sqrt(2 * delta * M * 3600)
        f = math.sqrt(2 * delta / (M * 3600))
        print(f"{delta:>6} s {t/60:>8.1f}min {f*100:>6.1f}% {t/120:>12.1f}min"
              f" {1440/(t/60):>8.0f}  {impl}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="all")
    ap.add_argument("rest", nargs="*")
    a = ap.parse_args()
    if a.cmd == "__worker__":
        fn, world, tmp, *extra = a.rest
        rank = int(extra[-1])
        extra = extra[:-1]
        if fn == "save":
            _worker_save(rank, int(world), tmp)
        elif fn == "reshard":
            _worker_reshard(rank, int(world), tmp, extra[0])
        elif fn == "async":
            _worker_async(rank, int(world), tmp)
        return
    tmp = tempfile.mkdtemp(prefix="ckpt08_")
    try:
        if a.cmd in ("all", "save", "reshard", "async"):
            print("===== save（4 rank）=====")
            _spawn("save", 4, tmp)
        if a.cmd in ("all", "reshard"):
            src = os.path.join(tmp, "ckpt")
            for w in (3, 2):
                print(f"\n===== reshard → {w} rank =====")
                _spawn("reshard", w, tmp, extra_args=[src])
        if a.cmd in ("all", "async"):
            print("\n===== async（2 rank）=====")
            _spawn("async", 2, tmp)
        if a.cmd in ("all", "young"):
            print("\n===== young =====")
            cmd_young(a)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
